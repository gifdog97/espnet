from collections import Counter
from math import log2
from pathlib import Path

import pandas as pd
from myutils import create_units_dict


def calc_entropy(units_list: list[list[str]]) -> float:
    all_units = [unit for sublist in units_list for unit in sublist]
    unit_counts = Counter(all_units)
    total_units = sum(unit_counts.values())
    entropy = 0.0
    for count in unit_counts.values():
        probability = count / total_units
        if probability > 0:
            entropy -= probability * log2(probability)
    return entropy


def calc_bitrate(units_list: list[list[str]]) -> float:
    entropy = calc_entropy(units_list)
    # https://keithito.com/LJ-Speech-Dataset/
    # Total duration of LJSpeech -> 23:55:17
    units_per_second = sum(len(u) for u in units_list) / (23 * 3600 + 55 * 60 + 17)
    return round(entropy * units_per_second, 1)


bitrate_dict = {}
for N in [20, 40, 80, 120, 160, 200, 240, 280]:
    for i in range(7, 15):
        K = 2**i
        setting = f"fixed_{N}-{K}_dedup"
        units_file = Path(
            f"/work/gk77/k77035/speechLM/experiment/units/LJSpeech-1.1/wavs/{setting}.csv"
        )
        units_dict = create_units_dict(units_file)
        units_list = [units.split(" ") for units in units_dict.values()]
        bitrate = calc_bitrate(units_list)
        keyN = f"N={N}"
        keyK = f"K={K}"
        if keyN not in bitrate_dict:
            bitrate_dict[keyN] = {}
        bitrate_dict[keyN][keyK] = bitrate

df = pd.DataFrame.from_dict(bitrate_dict)
df.to_csv("csv/bitrate.csv", index=True, sep="\t")
