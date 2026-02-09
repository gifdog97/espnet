import csv

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from matplotlib.colors import TwoSlopeNorm

rcParams["pdf.fonttype"] = 42
# フォントファイルのパスを指定
font_path = (
    "/work/01/gk77/k77035/.local/share/fonts/Times New Roman/times new roman.ttf"
)
mpl.rcParams["axes.unicode_minus"] = False


# フォントプロパティを作成
font_prop = fm.FontProperties(fname=font_path)

# グローバル設定に反映（全体に適用）
plt.rcParams["font.family"] = font_prop.get_name()


def extract_valid_settings(tts_model: str):
    valid_settings = []
    with open("csv/continuation_result.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["distance"]:
                continue
            setting = row["setting"]
            if not setting.startswith(tts_model):
                continue
            temperature = row["temperature"]
            bitrate = row["bitrate"]
            valid_settings.append(
                (f"{setting.split('-', 1)[1]}-{temperature}", bitrate)
            )
    return valid_settings


def extract_results(tts_model: str):
    if tts_model == "tacotron2":
        pairwise_result = "pairwise/result_summary.csv"
    elif tts_model == "vits":
        pairwise_result = "pairwise/result_summary-vits.csv"
    results_dict = {}
    with open(pairwise_result) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results_dict[f"{row['setting_X']}_{row['setting_Y']}"] = row[
                "average_score"
            ]
    return results_dict


def main():
    valid_settings_t = extract_valid_settings("tacotron2")
    valid_settings_v = extract_valid_settings("vits")
    results_dict_t = extract_results("tacotron2")
    results_dict_v = extract_results("vits")
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(8.2, 3.8),
        constrained_layout=True,
        dpi=300,
    )
    fig.get_layout_engine().set(hspace=0.10)  # ← 0.20 が既定。大きいほど行間が広がる
    ax = axes[0]
    ax.set_title("tacotron2", fontsize=16)
    data = []
    for setting_X, _ in valid_settings_t:
        heatmap_row = []
        for setting_Y, _ in valid_settings_t:
            score = results_dict_t.get(f"{setting_X}_{setting_Y}")
            if score is None:
                heatmap_row.append(0)
                continue
            heatmap_row.append(float(score))
        data.append(heatmap_row)
    data = np.array(data)
    # 0を白、負を赤、正を青にするカラーマップ
    cmap = plt.cm.bwr  # blue-white-red
    min_t, max_t = data.min(), data.max()
    norm = TwoSlopeNorm(vmin=min_t, vcenter=0, vmax=max_t)
    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        norm=norm,
        cbar=False,
        xticklabels=[
            f"{'-'.join(setting.split('-')[:-1])}" for (setting, _) in valid_settings_t
        ],
        yticklabels=[
            f"{'-'.join(setting.split('-')[:-1])} ({bitrate})"
            for (setting, bitrate) in valid_settings_t
        ],
    )
    ax.axhline(6, color="black", linewidth=2)
    ax.axhline(6 + 5, color="black", linewidth=2)
    ax.axhline(6 + 5 + 6, color="black", linewidth=2)
    ax.axvline(6, color="black", linewidth=2)
    ax.axvline(6 + 5, color="black", linewidth=2)
    ax.axvline(6 + 5 + 6, color="black", linewidth=2)

    ax = axes[1]
    ax.set_title("vits", fontsize=16)
    data = []
    for setting_X, _ in valid_settings_v:
        heatmap_row = []
        for setting_Y, _ in valid_settings_v:
            score = results_dict_v.get(f"{setting_X}_{setting_Y}")
            if score is None:
                heatmap_row.append(0)
                continue
            heatmap_row.append(float(score))
        data.append(heatmap_row)
    data = np.array(data)
    # 0を白、負を赤、正を青にするカラーマップ
    cmap = plt.cm.bwr  # blue-white-red
    norm = TwoSlopeNorm(vmin=min_t, vcenter=0, vmax=max_t)
    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        norm=norm,
        cbar=True,
        xticklabels=[
            f"{'-'.join(setting.split('-')[:-1])}" for (setting, _) in valid_settings_v
        ],
        yticklabels=[
            f"{'-'.join(setting.split('-')[:-1])} ({bitrate})"
            for (setting, bitrate) in valid_settings_v
        ],
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    ax.axhline(7, color="black", linewidth=2)
    ax.axhline(7 + 7, color="black", linewidth=2)
    ax.axvline(7, color="black", linewidth=2)
    ax.axvline(7 + 7, color="black", linewidth=2)

    plt.savefig("fig/pairwise_heatmap.pdf")


if __name__ == "__main__":
    main()
