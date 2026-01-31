import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

CONTINUATION_RESULTS = "/work/gk77/k77035/espnet/egs2/ljspeech/tts1/myscripts/csv/continuation_result.tsv"  # CHANGE

pairwise_result = "pairwise/result_summary.csv"  # CHANGE


def main():
    with open(Path(CONTINUATION_RESULTS)) as f:
        reader = csv.DictReader(f, delimiter="\t")
        valid_settings = [
            f"{row['setting']}-{row['temperature']}"
            for row in reader
            if row["temperature"] != "None"
        ]
    print(valid_settings)
    results_dict = {}
    with open(pairwise_result) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results_dict[f"{row['setting_X']}_{row['setting_Y']}"] = row[
                "average_score"
            ]
    data = []
    for setting_X in valid_settings:
        heatmap_row = []
        for setting_Y in valid_settings:
            score = results_dict.get(f"{setting_X}_{setting_Y}")
            if score is None:
                heatmap_row.append(0)
                continue
            heatmap_row.append(float(score))
        data.append(heatmap_row)
    data = np.array(data)
    # 0を白、負を赤、正を青にするカラーマップ
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.bwr  # blue-white-red
    norm = TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())
    sns.heatmap(
        data,
        cmap=cmap,
        norm=norm,
        cbar=True,
        xticklabels=["-".join(setting.split("-")[:-1]) for setting in valid_settings],
        yticklabels=["-".join(setting.split("-")[:-1]) for setting in valid_settings],
    )
    # CHANGE
    ax.axhline(6, color="black", linewidth=2)
    ax.axhline(6 + 5, color="black", linewidth=2)
    ax.axhline(6 + 5 + 7, color="black", linewidth=2)
    ax.axhline(6 + 5 + 7 + 7, color="black", linewidth=2)
    ax.axvline(6, color="black", linewidth=2)
    ax.axvline(6 + 5, color="black", linewidth=2)
    ax.axvline(6 + 5 + 7, color="black", linewidth=2)
    ax.axvline(6 + 5 + 7 + 7, color="black", linewidth=2)
    # ax.axhline(8, color="black", linewidth=2)
    # ax.axhline(8 + 8, color="black", linewidth=2)
    # ax.axhline(8 + 8 + 8, color="black", linewidth=2)
    # ax.axhline(8 + 8 + 8 + 8, color="black", linewidth=2)
    # ax.axvline(8, color="black", linewidth=2)
    # ax.axvline(8 + 8, color="black", linewidth=2)
    # ax.axvline(8 + 8 + 8, color="black", linewidth=2)
    # ax.axvline(8 + 8 + 8 + 8, color="black", linewidth=2)

    plt.savefig("fig/pairwise_heatmap.png")  # CHANGE


if __name__ == "__main__":
    main()
