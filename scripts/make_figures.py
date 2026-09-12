"""Vỏ mỏng cho ``reporting.figures`` — logic nằm trong module đã có test.

    python scripts/make_figures.py
"""

import argparse

from reporting.figures import make_all_figures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args()

    for path in make_all_figures(args.results_dir, args.out_dir):
        print(f"  {path}  ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
