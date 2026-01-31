import math

import japanize_matplotlib  # noqa: F401
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams

rcParams["pdf.fonttype"] = 42
# フォントファイルのパスを指定
font_path = (
    "/work/01/gk77/k77035/.local/share/fonts/Times New Roman/times new roman.ttf"
)

# フォントプロパティを作成
font_prop = fm.FontProperties(fname=font_path)

# グローバル設定に反映（全体に適用）
plt.rcParams["font.family"] = font_prop.get_name()


def plot(axes, df, num):
    # prepare data
    value_dict = {}
    for idx, row in df.iterrows():
        N, K = idx.split("-")
        if N not in value_dict:
            value_dict[N] = {"K": [], "bitrate": [], "PPL": [], "VERT": []}
        value_dict[N]["bitrate"].append(row["bitrate"])
        value_dict[N]["K"].append(K)
        value_dict[N]["PPL"].append(row["PPL"])
        value_dict[N]["VERT"].append(row["VERT"])

    # plot PPL
    ax = axes[0][num]
    ax.grid()
    if num == 0:
        ax.set_title("Tacotron2", fontsize=12)
    if num == 1:
        ax.set_title("VITS", fontsize=12)
    for setting, values in value_dict.items():
        N = setting.split("-")[0]
        x = [int(math.log2(int(K)) - 6) for K in values["K"]]
        y = values["PPL"]
        ax.plot(
            x, y, marker="o", markersize=2.5, linewidth=2, alpha=0.7, label=f"N={N}"
        )
    # oracle line
    ax.axhline(52.264, color="black", linestyle="--", label="Oracle")
    if num == 0:
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7], minor=False)
    elif num == 1:
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8], minor=False)
    ax.set_xticklabels([])  # 上のグラフのx軸ラベルを消す
    ax.set_ylim(0, 400)
    ax.set_yticks([0, 100, 200, 300, 400])
    if num == 0:
        ax.set_ylabel(r"PPL$\downarrow$", fontsize=12)
    if num == 1:
        ax.set_yticklabels([])

    # plot VERT
    ax = axes[1][num]
    ax.grid()
    for setting, values in value_dict.items():
        N = setting.split("-")[0]
        x = [int(math.log2(int(K)) - 6) for K in values["K"]]
        y = values["VERT"]
        ax.plot(
            x, y, marker="o", markersize=2.5, linewidth=2, alpha=0.7, label=f"N={N}"
        )
    # oracle line
    ax.axhline(10.47, color="black", linestyle="--", label="Oracle")
    if num == 0:
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7], minor=False)
        ax.set_xticklabels(
            [
                r"$2^7$",
                r"$2^8$",
                r"$2^9$",
                r"$2^{10}$",
                r"$2^{11}$",
                r"$2^{12}$",
                r"$2^{13}$",
            ],
            minor=False,
        )
    elif num == 1:
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8], minor=False)
        ax.set_xticklabels(
            [
                r"$2^7$",
                r"$2^8$",
                r"$2^9$",
                r"$2^{10}$",
                r"$2^{11}$",
                r"$2^{12}$",
                r"$2^{13}$",
                r"$2^{14}$",
            ],
            minor=False,
        )
    if num == 0:
        ax.set_ylabel(r"VERT$\downarrow$", fontsize=12)
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1, 0),
            fontsize=8,
            ncol=2,
        )
    if num == 1:
        ax.set_yticklabels([])
    ax.set_ylim(5, 28)
    ax.set_yticks([10, 15, 20, 25])
    ax.set_xlabel("Cluster size (K)", fontsize=12)


if __name__ == "__main__":
    # load results of tacotron2 and vits
    df_t = pd.read_csv("./csv/continuation_result.tsv", sep="\t", index_col=0)
    df_v = pd.read_csv("./csv/continuation_result-vits.tsv", sep="\t", index_col=0)
    # filter out ill-performed settings
    df_t = df_t[df_t["temperature"].notna()]
    df_v = df_v[df_v["temperature"].notna()]

    # prepare for plotting
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(4.5, 3.8),
        constrained_layout=True,
        dpi=300,
    )
    fig.get_layout_engine().set(hspace=0.10)  # ← 0.20 が既定。大きいほど行間が広がる

    plot(axes, df_t, 0)
    plot(axes, df_v, 1)

    fig.savefig("./fig/continuation_score_K.pdf")
