import argparse
import math

import japanize_matplotlib  # noqa: F401
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams

# ======================
# Matplotlib global setup
# ======================
rcParams["pdf.fonttype"] = 42

font_path = (
    "/work/01/gk77/k77035/.local/share/fonts/Times New Roman/times new roman.ttf"
)
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()

markers = "osDv^<>X"


# ======================
# Utilities
# ======================
def prepare_values(df, x_axis: str):
    """
    Returns:
        dict[N] = {
            "x": [...],
            "PPL": [...],
            "VERT": [...]
        }
    """
    value_dict = {}

    for idx, row in df.iterrows():
        parts = idx.split("-")
        N = parts[0]

        if N not in value_dict:
            value_dict[N] = {"x": [], "PPL": [], "VERT": []}

        if x_axis == "K":
            K = int(parts[1])
            x = int(math.log2(K) - 6)
        elif x_axis == "bitrate":
            x = row["bitrate"]
        else:
            raise ValueError(f"Unknown x_axis: {x_axis}")

        value_dict[N]["x"].append(x)
        value_dict[N]["PPL"].append(row["PPL"])
        value_dict[N]["VERT"].append(row["VERT"])

    return value_dict


def configure_xaxis(ax, num: int, x_axis: str):
    if x_axis == "K":
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
    elif x_axis == "bitrate":
        if num == 0:
            ax.set_xlim(0, 500)
            ticks = [0, 100, 200, 300, 400, 500]
        else:
            ax.set_xlim(0, 600)
            ticks = [0, 100, 200, 300, 400, 500, 600]

        ax.set_xticks(ticks)
        ax.set_xticklabels([t // 100 for t in ticks])


def xlabel(x_axis: str):
    if x_axis == "K":
        return "Cluster size (K)"
    elif x_axis == "bitrate":
        return "Bitrate (x100)"


# ======================
# Plotting
# ======================
def plot(axes, df, num: int, x_axis: str):
    values = prepare_values(df, x_axis)

    # ---- PPL ----
    ax = axes[0][num]
    ax.grid()

    ax.set_title("Tacotron2" if num == 0 else "VITS", fontsize=12)

    for i, (N, v) in enumerate(values.items()):
        ax.plot(
            v["x"],
            v["PPL"],
            marker=markers[i % len(markers)],
            markersize=3 if x_axis == "bitrate" else 2.5,
            linewidth=2,
            alpha=0.7,
            label=f"N={N}",
        )

    ax.axhline(52.264, color="black", linestyle="--", label="Oracle")
    ax.set_ylim(0, 400)
    ax.set_yticks([0, 100, 200, 300, 400])

    if num == 0:
        ax.set_ylabel(r"PPL$\downarrow$", fontsize=12)
    else:
        ax.set_yticklabels([])

    ax.set_xticklabels([])

    # ---- VERT ----
    ax = axes[1][num]
    ax.grid()

    for i, (N, v) in enumerate(values.items()):
        ax.plot(
            v["x"],
            v["VERT"],
            marker=markers[i % len(markers)],
            markersize=3 if x_axis == "bitrate" else 2.5,
            linewidth=2,
            alpha=0.7,
            label=f"N={N}",
        )

    ax.axhline(10.47, color="black", linestyle="--", label="Oracle")
    ax.set_ylim(5, 28)
    ax.set_yticks([10, 15, 20, 25])

    configure_xaxis(ax, num, x_axis)

    if num == 0:
        ax.set_ylabel(r"VERT$\downarrow$", fontsize=12)
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1, 0),
            fontsize=8,
            ncol=2,
        )
    else:
        ax.set_yticklabels([])

    ax.set_xlabel(xlabel(x_axis), fontsize=12)


# ======================
# Main
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--x-axis",
        choices=["K", "bitrate"],
        default="K",
        help="x-axis type",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output pdf path",
    )
    args = parser.parse_args()

    # load results
    df_t = pd.read_csv("./csv/continuation_result.tsv", sep="\t", index_col=0)
    df_v = pd.read_csv("./csv/continuation_result-vits.tsv", sep="\t", index_col=0)

    df_t = df_t[df_t["temperature"].notna()]
    df_v = df_v[df_v["temperature"].notna()]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(4.5, 3.8),
        constrained_layout=True,
        dpi=300,
    )
    fig.get_layout_engine().set(hspace=0.10)

    plot(axes, df_t, 0, args.x_axis)
    plot(axes, df_v, 1, args.x_axis)

    if args.output is None:
        suffix = "K" if args.x_axis == "K" else "bitrate"
        args.output = f"./fig/continuation_score_{suffix}.pdf"

    fig.savefig(args.output)
