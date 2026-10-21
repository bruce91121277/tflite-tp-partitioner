"""Run TP2 or TP4 partitioning on a user-supplied TFLite graph."""

import argparse
from pathlib import Path

from tflite_tp.cli import main


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="authorized source TFLite file or directory")
    parser.add_argument("output", type=Path, help="new output directory")
    parser.add_argument("--tp", choices=("2", "4"), default="4")
    parser.add_argument("--profile", choices=("D1", "P128"), default="D1")
    parser.add_argument("--expected-layers", type=int, default=1)
    return parser.parse_args()


def main_example():
    args = parse_args()
    command = [
        "build-tp{}".format(args.tp),
        "--source",
        str(args.source),
        "--output",
        str(args.output),
        "--profile",
        args.profile,
        "--expected-layers",
        str(args.expected_layers),
    ]
    return main(command)


if __name__ == "__main__":
    raise SystemExit(main_example())