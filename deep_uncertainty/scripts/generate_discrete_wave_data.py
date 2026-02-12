from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np

from deep_uncertainty.data_generator import DataGenerator


def main(n: int, out_path: Path, seed: int = 0) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    splits = DataGenerator.generate_train_val_test_split(
        data_gen_function=DataGenerator.generate_discrete_conflation_sine_wave,
        data_gen_params={"n": int(n)},
        random_seed=int(seed),
    )
    np.savez(out_path, **splits)
    print(f"Saved dataset to: {out_path}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of (x, y) samples to generate before splitting.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/discrete-wave/discrete_sine_wave.npz",
        help="Output .npz path for train/val/test split.",
    )
    parser.add_argument("--seed", type=int, default=1998, help="Random seed for splitting.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(n=args.n, out_path=Path(args.out), seed=args.seed)

