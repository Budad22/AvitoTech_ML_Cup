"""Four-hour adaptive research queue, waiting without a language-model connection."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
import torch

from ftt.data import Cache, digest, prepare, write_json
from ftt.model import Ranker, configuration
from ftt.runner import Queue
from ftt.training import evaluate
from ftt.policy import rank_result, select_candidates, validation_partition, paired_summary
from ftt.runtime import read_json, worker_lock, record_failure, add_run_arguments


def gpu_busy(predecessors=('FTT-NIGHT-001','FTT-NIGHT-002')):
    # Do not compete with the preceding FTT process, even during CPU evaluation.
    import re
    if any(not re.fullmatch(r'FTT-[A-Z0-9-]+',name) for name in predecessors):
        raise ValueError('Invalid predecessor run ID')
    pattern='(?:'+'|'.join(predecessors)+').*bootstrap'
    command = "@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and $_.CommandLine -match '"+pattern+"'}).Count"
    count = subprocess.check_output(['powershell','-NoProfile','-NonInteractive','-Command',command],text=True)
    if int(count.strip()):
        return True
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used',
                                    '--format=csv,noheader,nounits'],text=True).strip().splitlines()
    return any(float(row.split(',')[0]) > 10 or float(row.split(',')[1]) > 1000 for row in rows)


def import_control(root, source, control=9, frozen_budget=None):
    """Copy only after the preceding writer has exited; never overwrite a resumed run."""
    root,source=Path(root),Path(source)
    if (root/'ledger.json').exists():
        return
    prior=json.loads((source/'ledger.json').read_text(encoding='utf-8'))
    ledger={}
    copied={}
    for key,result in prior.items():
        phase=key.split('/')[0]
        if result['number']!=control or phase not in ('smoke','small','medium'):
            continue
        if phase=='medium' and frozen_budget and result['target']>frozen_budget:
            continue
        relative=Path(phase)/f"FTT-{control:03d}-s{result['seed']}"
        destination=root/relative
        if str(relative) not in copied:
            shutil.copytree(source/relative,destination,dirs_exist_ok=True)
            if phase=='medium' and frozen_budget:
                frozen=destination/'confirmed-best.pt'
                if not frozen.exists():
                    raise ValueError('Frozen control checkpoint is required')
                shutil.copy2(frozen,destination/'best.pt')
                shutil.copy2(frozen,destination/'last.pt')
            copied[str(relative)]={p.name:digest(p) for p in destination.glob('*.pt')}
        ledger[key]=dict(result,directory=str(destination))
    if not ledger:
        raise ValueError('Predecessor has no reusable control')
    write_json(root/'control-source.json',dict(source=str(source),files=copied,control=control,frozen_budget=frozen_budget))
    write_json(root/'ledger.json',ledger)


class Followup(Queue):
    def remaining(self):
        return self.args.hours*3600-self.previous_seconds-(time.monotonic()-self.started)

    def snapshot(self, **values):
        self.report.update(values)
        self.report['completed_trials'] = self.ledger
        write_json(self.root/'report.json',self.report)

    def execute(self, number, seed, target, size):
        result = self.trial(number,seed,target,size)
        self.snapshot(phase=size)
        return result

    def screen(self, number, target, size):
        if str(number) in self.excluded:
            return None
        try:
            return self.execute(number,42,target,size)
        except (ValueError,RuntimeError) as error:
            recoverable = isinstance(error,torch.cuda.OutOfMemoryError) or 'nonfinite' in str(error).lower() or 'non-finite' in str(error).lower()
            if number==self.control or not recoverable:
                raise
            self.excluded[str(number)]=traceback.format_exc()
            write_json(self.root/'excluded.json',self.excluded)
            self.emit('excluded',model=f'FTT-{number:03d}',error=str(error))
            return None

    def run(self):
        self.control=getattr(self.args,'control',9)
        self.report = dict(experiment=self.args.experiment, status='running', control=self.control,
                           full_training='disabled', confirmation_reused=True)
        info = prepare(self.args.data,self.root/'cache',self.emit)
        self.validation = Cache(self.root/'cache','val')
        rng = np.random.default_rng(2026)
        selection, self.fast, confirmation = validation_partition(self.validation.timestamps, info['boundary'], rng)
        self.excluded = read_json(self.root/'excluded.json', {})
        variants = [int(n) for n in getattr(self.args,'variants','13,14,15,16,17,18,19,20').split(',')]
        if self.control in variants or len(set(variants))!=len(variants) or any(n not in range(1,27) for n in variants):
            raise ValueError('Hypotheses must be unique IDs 1..26 excluding the control')
        for number in variants:
            self.screen(number,20,'smoke')
        for target in (200,800):
            screened = [r for n in [self.control,*variants] if (r:=self.screen(n,target,'small'))]
        candidates = sorted([r for r in screened if r['number']!=self.control],key=rank_result)
        if not candidates:
            self.snapshot(status='inconclusive',reason='No viable hypotheses')
            self.emit('completed',status='completed')
            return
        selected = select_candidates(screened, control=self.control, top_count=3)[1:]
        extended = [r for n in [self.control,*selected] if (r:=self.screen(n,3200,'small'))]
        finalists = [r['number'] for r in sorted([r for r in extended if r['number']!=self.control],key=rank_result)[:2]]
        self.snapshot(finalists=finalists,phase='medium')
        medium = [r for n in [self.control,*finalists] if (r:=self.screen(n,12800,'medium'))]
        if not any(r['number']!=self.control for r in medium):
            self.snapshot(status='inconclusive',reason='No viable medium hypotheses')
            self.emit('completed',status='completed')
            return
        winner = min([r for r in medium if r['number']!=self.control],key=rank_result)['number']
        self.snapshot(winner=winner)
        for seed in (43,44):
            for n in (self.control,winner):
                self.execute(n,seed,12800,'medium')
        metrics, checkpoints = {}, []
        statistics = Cache(self.root/'cache','train','medium').statistics()
        for n in (self.control,winner):
            metrics[n]=[]
            for seed in (42,43,44):
                result=self.execute(n,seed,12800,'medium')
                folder=Path(result['directory'])
                frozen=folder/'confirmed-best.pt'
                if not frozen.exists():
                    shutil.copy2(folder/'best.pt',frozen)
                state=torch.load(frozen,map_location='cpu',weights_only=False)
                model=Ranker(configuration(n),*statistics).to(self.args.device)
                model.load_state_dict(state['model'])
                self.emit('confirmation',phase='confirmation',model=f'FTT-{n:03d}',seed=seed)
                values=evaluate(model,self.validation,confirmation,self.args.device)
                np.save(folder/'confirmation-12800.npy',values)
                metrics[n].append(values)
                checkpoints.append(dict(number=n,seed=seed,step=state['step'],
                                        sha256=digest(frozen),path=str(frozen)))
        summary = paired_summary(metrics[winner], metrics[self.control], self.validation.users[confirmation], rng)
        self.snapshot(phase='convergence',confirmation=dict(control=float(np.mean(metrics[self.control])),
                      winner=float(np.mean(metrics[winner])),delta=summary['delta'],
                      per_seed=summary['per_seed'],interval=summary['interval'],
                      checkpoints=checkpoints,feeds=len(confirmation)))
        # Freeze confirmed checkpoints before exploring longer seed42 curves.
        for target in (25600,51200):
            previous=12800 if target==25600 else 25600
            estimates=[]
            for n in (self.control,winner):
                result=self.ledger[f'medium/FTT-{n:03d}-s42/{previous}']
                estimates.append(result['seconds']/result['steps']*(target-previous))
            if self.remaining() < sum(estimates)*1.35+300:
                self.snapshot(long_extension_skipped=target,reason='Insufficient budget for both matched curves')
                break
            for n in (self.control,winner):
                self.execute(n,42,target,'medium')
            self.snapshot(longest_matched_steps=target,
                          long_scope='Exploratory seed42 curves; confirmation above uses frozen 12800-step budgets')
        self.snapshot(status='completed',phase='completed')
        self.emit('completed',status='completed',phase='completed')


def main():
    parser=argparse.ArgumentParser()
    add_run_arguments(parser, hours=4, experiment='FTT-NIGHT-003')
    parser.add_argument('--control-source',default='Q:/AVITOCUP/experiments/FTT-NIGHT-002')
    parser.add_argument('--control',type=int,default=9)
    parser.add_argument('--variants',default='13,14,15,16,17,18,19,20')
    parser.add_argument('--frozen-control-budget',type=int)
    parser.add_argument('--wait-for',default='FTT-NIGHT-001,FTT-NIGHT-002')
    args=parser.parse_args()
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    with worker_lock(root/'followup.lock'):
        queued=time.monotonic()
        prior=read_json(root/'status.json', {})
        elapsed=prior.get('elapsed_seconds',0)
        while gpu_busy(tuple(args.wait_for.split(','))):
            write_json(root/'status.json',dict(experiment=args.experiment,status='waiting',
                       phase='waiting_for_gpu',pid=os.getpid(),heartbeat=time.time(),
                       elapsed_seconds=elapsed,budget_seconds=args.hours*3600,waiting_seconds=time.monotonic()-queued))
            time.sleep(15)
        # Tests use CUDA only after the preceding queue has released the GPU.
        with (root/'tests.log').open('a') as out, (root/'tests.err.log').open('a') as err:
            tests=subprocess.run([os.sys.executable,'-X','utf8','-m','unittest','discover','-s','tests','-v'],
                                 stdout=out,stderr=err)
        if tests.returncode:
            write_json(root/'status.json',dict(experiment=args.experiment,status='failed',
                       error='CUDA preflight failed; see tests.err.log',pid=os.getpid(),heartbeat=time.time()))
            return
        import_control(root,args.control_source,args.control,args.frozen_control_budget)
        queue=Followup(args)
        try:
            queue.run()
        except Exception as error:
            record_failure(queue, error, queue.snapshot if hasattr(queue,'report') else None)
            raise


if __name__=='__main__':
    main()
