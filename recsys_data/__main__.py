import argparse
from pathlib import Path

from .core import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description="Validate and prepare model-independent Avito feed data")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--allow-missing-embeddings", action="store_true",
                        help="Prepare tables even when a matrix is not downloaded; vector lookup will fail explicitly")
    parser.add_argument("--archive-manifest", type=Path)
    args = parser.parse_args()
    prepare_dataset(args.data, args.output, args.splits, batch_size=args.batch_size,
                    allow_missing_embeddings=args.allow_missing_embeddings,
                    archive_manifest=args.archive_manifest)


if __name__ == "__main__":
    main()
