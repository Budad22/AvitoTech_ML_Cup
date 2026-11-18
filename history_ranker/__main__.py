import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="EXP-TRF-003 candidate-to-history Transformer")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--raw", required=True, type=Path)
    prepare.add_argument("--compact", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--scope", choices=("smoke", "full"), required=True)
    for name in ("smoke", "pilot", "train-a", "train-b", "evaluate", "evaluate-pilot"):
        command = commands.add_parser(name)
        command.add_argument("--cache", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        if name != "smoke":
            command.add_argument("--microbatch", type=int, choices=(1, 2, 4, 8), default=8)
        if name in ("pilot", "train-a", "train-b"):
            command.add_argument("--query-chunk", type=int, choices=(32, 64, 128))
        if name in ("train-b", "evaluate"):
            command.add_argument("--stage-a", required=True, type=Path)
        if name == "evaluate":
            for key in ("stage-b", "baseline", "official"):
                command.add_argument("--" + key, required=True, type=Path)
        if name == "evaluate-pilot":
            command.add_argument("--pilot", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "prepare":
        from .prepare import prepare as build
        build(args.raw, args.compact, args.output, args.scope)
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.command == "smoke":
            from .train import smoke
            smoke(args.cache, args.output, args.device)
        elif args.command.startswith("evaluate"):
            from .evaluate import evaluate
            evaluate(args)
        else:
            from .train import train
            stage = args.command.removeprefix("train-") if args.command != "pilot" else "pilot"
            train(args.cache, args.output, stage, args.microbatch, args.device,
                  parent=getattr(args, "stage_a", None), query_chunk=args.query_chunk)


if __name__ == "__main__":
    main()
