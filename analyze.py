import argparse
import numpy as np
from prettytable import PrettyTable

from utils import *

labels = ["generate", "acc_len"]


def process(raw: dict):
    stat = {label: [np.mean(r[label]) for r in raw] for label in labels}
    return stat


def output(target: dict, target_name: str, base: dict):
    assert len(base) == len(target)

    print(f"[{target_name}]")

    tab = PrettyTable(["Label", "Mean", "Min", "Max", "Std", "Mean/BaseMean"])
    tab.align = "r"

    for label in labels:
        s = target[label]
        tab.add_row(
            [
                label,
                np.mean(s),
                np.min(s),
                np.max(s),
                np.std(s),
                np.mean(s) / np.mean(base[label]),
            ]
        )
    print(tab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files", nargs="+", help="stat file, the first will be used as base"
    )
    args = parser.parse_args()

    stat = [process(load_jsonl(file)) for file in args.files]

    for tgt, name in zip(stat, args.files):
        output(tgt, name, stat[0])
