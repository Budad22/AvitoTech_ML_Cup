"""Four-hour adaptive research queue, waiting without a language-model connection."""
import argparse
import ctypes
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
from ftt.runner import Queue, evaluate


def gpu_busy():
    # Do not compete with the preceding FTT process, even during CPU evaluation.
    command = "@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and $_.CommandLine -match 'FTT-NIGHT-00[12].*bootstrap'}).Count"
    count = subprocess.check_output(['powershell','-NoProfile','-NonInteractive','-Command',command],text=True)
    if int(count.strip()):
        return True
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used',
                                    '--format=csv,noheader,nounits'],text=True).strip().splitlines()
    return any(float(row.split(',')[0]) > 10 or float(row.split(',')[1]) > 1000 for row in rows)


def import_control(root, source):
    """Copy only after the preceding writer has exited; never overwrite a resumed run."""
    root,source=Path(root),Path(source)
    if (root/'ledger.json').exists():
        return
    prior=json.loads((source/'ledger.json').read_text(encoding='utf-8'))
    ledger={}
    copied={}
    for key,result in prior.items():
        phase=key.split('/')[0]
        if result['number']!=9 or phase not in ('smoke','small','medium'):
            continue
        relative=Path(phase)/f"FTT-009-s{result['seed']}"
        destination=root/relative
        if str(relative) not in copied:
            shutil.copytree(source/relative,destination,dirs_exist_ok=True)
            copied[str(relative)]={p.name:digest(p) for p in destination.glob('*.pt')}
        ledger[key]=dict(result,directory=str(destination))
    if not ledger:
        raise ValueError('Predecessor has no reusable FTT-009 control')
    write_json(root/'control-source.json',dict(source=str(source),files=copied))
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
            if number==9 or not recoverable:
                raise
            self.excluded[str(number)]=traceback.format_exc()
            write_json(self.root/'excluded.json',self.excluded)
            self.emit('excluded',model=f'FTT-{number:03d}',error=str(error))
            return None

    def run(self):
        self.report = dict(experiment=self.args.experiment, status='running', control=9,
                           full_training='disabled', confirmation_reused=True)
        info = prepare(self.args.data,self.root/'cache',self.emit)
        self.validation = Cache(self.root/'cache','val')
        selection = np.flatnonzero(self.validation.timestamps < info['boundary'])
        rng = np.random.default_rng(2026)
        self.fast = np.sort(rng.permutation(selection)[:min(2000,len(selection)//2)])
        confirmation = np.setdiff1d(selection,self.fast)
        self.excluded = json.loads((self.root/'excluded.json').read_text()) if (self.root/'excluded.json').exists() else {}
        variants = list(range(13,21))
        for number in variants:
            self.screen(number,20,'smoke')
        for target in (200,800):
            screened = [r for n in [9,*variants] if (r:=self.screen(n,target,'small'))]
        rank = lambda r:(-r['best'],r['seconds'],r['number'])
        candidates = sorted([r for r in screened if r['number']!=9],key=rank)
        if not candidates:
            self.snapshot(status='inconclusive',reason='No viable hypotheses')
            self.emit('completed',status='completed')
            return
        # Reserve a place for a slowly improving variant.
        selected = [r['number'] for r in candidates[:3]]
        def growth(r):
            h={v['step']:v['ndcg'] for v in r['history']}
            return (-(h[800]-h[200]),r['number'])
        remaining=[r for r in candidates if r['number'] not in selected]
        if remaining:selected.append(min(remaining,key=growth)['number'])
        extended = [r for n in [9,*selected] if (r:=self.screen(n,3200,'small'))]
        finalists = [r['number'] for r in sorted([r for r in extended if r['number']!=9],key=rank)[:2]]
        self.snapshot(finalists=finalists,phase='medium')
        medium = [r for n in [9,*finalists] if (r:=self.screen(n,12800,'medium'))]
        if not any(r['number']!=9 for r in medium):
            self.snapshot(status='inconclusive',reason='No viable medium hypotheses')
            self.emit('completed',status='completed')
            return
        winner = min([r for r in medium if r['number']!=9],key=rank)['number']
        self.snapshot(winner=winner)
        for seed in (43,44):
            for n in (9,winner):
                self.execute(n,seed,12800,'medium')
        metrics, checkpoints = {}, []
        statistics = Cache(self.root/'cache','train','medium').statistics()
        for n in (9,winner):
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
        delta=np.asarray(metrics[winner])-np.asarray(metrics[9])
        _,groups,counts=np.unique(self.validation.users[confirmation],return_inverse=True,return_counts=True)
        sums=np.stack([np.bincount(groups,weights=row) for row in delta])
        draws=[]
        for _ in range(20000):
            sampled_seeds=rng.integers(0,3,3)
            sampled_users=rng.integers(0,len(counts),len(counts))
            draws.append(sums[sampled_seeds].mean(0)[sampled_users].sum()/counts[sampled_users].sum())
        lower,upper=np.quantile(draws,[.025,.975])
        self.snapshot(phase='convergence',confirmation=dict(control=float(np.mean(metrics[9])),
                      winner=float(np.mean(metrics[winner])),delta=float(delta.mean()),
                      per_seed=delta.mean(1).tolist(),interval=[float(lower),float(upper)],
                      checkpoints=checkpoints,feeds=len(confirmation)))
        # Freeze confirmed checkpoints before exploring longer seed42 curves.
        for target in (25600,51200):
            previous=12800 if target==25600 else 25600
            estimates=[]
            for n in (9,winner):
                result=self.ledger[f'medium/FTT-{n:03d}-s42/{previous}']
                estimates.append(result['seconds']/result['steps']*(target-previous))
            if self.remaining() < sum(estimates)*1.35+300:
                self.snapshot(long_extension_skipped=target,reason='Insufficient budget for both matched curves')
                break
            for n in (9,winner):
                self.execute(n,42,target,'medium')
            self.snapshot(longest_matched_steps=target,
                          long_scope='Exploratory seed42 curves; confirmation above uses frozen 12800-step budgets')
        self.snapshot(status='completed',phase='completed')
        self.emit('completed',status='completed',phase='completed')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--hours',type=float,default=4)
    parser.add_argument('--experiment',default='FTT-NIGHT-003')
    parser.add_argument('--control-source',default='Q:/AVITOCUP/experiments/FTT-NIGHT-002')
    args=parser.parse_args()
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    with (root/'followup.lock').open('a+b') as lock:
        import msvcrt
        lock.seek(0)
        if not lock.read(1):lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
        queued=time.monotonic()
        prior=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
        elapsed=prior.get('elapsed_seconds',0)
        while gpu_busy():
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
        import_control(root,args.control_source)
        queue=Followup(args)
        try:
            queue.run()
        except Exception as error:
            (root/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
            status='budget_exhausted' if isinstance(error,TimeoutError) else 'failed'
            if hasattr(queue,'report'):queue.snapshot(status=status,error=str(error))
            queue.state.update(status=status,error=str(error),heartbeat=time.time())
            write_json(root/'status.json',queue.state)
            raise


if __name__=='__main__':
    main()
