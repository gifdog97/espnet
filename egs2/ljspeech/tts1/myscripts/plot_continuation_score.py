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
        dict[model-N] = {
            "x": [...],
            "PPL": [...],
            "VERT": [...]
        }
    """
    value_dict = {}
    for idx, row in df.iterrows():
        if idx == "gold":
            value_dict["gold"] = {
                "x": [0],
                "PPL": [row["PPL"]],
                "VERT": [row["VERT"]],
            }
            continue
        parts = idx.split("-")
        setting = parts[0] + "-" + parts[1]  #
        if setting not in value_dict:
            value_dict[setting] = {"x": [], "PPL": [], "VERT": []}

        if x_axis == "K":
            K = int(parts[2])
            x = int(math.log2(K) - 6)
        elif x_axis == "bitrate":
            x = row["bitrate"]
        else:
            raise ValueError(f"Unknown x_axis: {x_axis}")

        value_dict[setting]["x"].append(x)
        value_dict[setting]["PPL"].append(row["PPL"])
        value_dict[setting]["VERT"].append(row["VERT"])

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
        ax.set_xlim(80, 520)
        ticks = [100, 200, 300, 400, 500]

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
def plot(axes, df, x_axis: str):
    values = prepare_values(df, x_axis)

    for num in range(2):
        # ---- PPL ----
        ax = axes[0][num]
        ax.grid()

        title = "tacotron2" if num == 0 else "vits"
        ax.set_title(title, fontsize=12)

        plot_idx = 0
        for setting, v in values.items():
            if setting == "gold":
                continue
            model, N = setting.split("-")
            if model != title:
                continue
            ax.plot(
                v["x"],
                v["PPL"],
                marker=markers[plot_idx],
                markersize=3,
                linewidth=2,
                alpha=0.7,
                label=f"N={N}",
            )
            plot_idx += 1

        ax.axhline(
            values["gold"]["PPL"][0], color="black", linestyle="--", label="Gold"
        )
        ylabels = [0, 50, 100, 150, 200, 250, 300]
        ax.set_ylim(0, 300)
        ax.set_yticks(ylabels)

        configure_xaxis(ax, num, x_axis)

        if num == 0:
            ax.set_ylabel(r"PPL$\downarrow$", fontsize=10)
            ax.set_yticklabels(ylabels, fontsize=9)
        else:
            ax.set_yticklabels([])

        ax.set_xticklabels([])

        # ---- VERT ----
        ax = axes[1][num]
        ax.grid()

        plot_idx = 0
        for setting, v in values.items():
            if setting == "gold":
                continue
            model, N = setting.split("-")
            if model != title:
                continue
            ax.plot(
                v["x"],
                v["VERT"],
                marker=markers[plot_idx],
                markersize=3,
                linewidth=2,
                alpha=0.7,
                label=f"N={N}",
            )
            plot_idx += 1

        ax.axhline(
            values["gold"]["VERT"][0], color="black", linestyle="--", label="Gold"
        )
        ax.set_ylim(7, 27)
        ylabels = [10, 15, 20, 25]
        ax.set_yticks(ylabels)

        configure_xaxis(ax, num, x_axis)

        if num == 0:
            ax.set_ylabel(r"VERT$\downarrow$", fontsize=10)
            ax.set_yticklabels(ylabels, fontsize=9)
            bbox = ax.get_position()
            hans, labs = ax.get_legend_handles_labels()
            fig.legend(
                handles=hans,
                labels=labs,
                loc="lower left",
                bbox_to_anchor=(bbox.x0, bbox.y0 + 0.03),
                fontsize=8,
                ncol=3,
            )
        else:
            ax.set_yticklabels([])

        ax.set_xlabel(xlabel(x_axis), fontsize=10)


# ======================
# Main
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--continuation-result-csv",
        default="./csv/continuation_result_10s.csv",
        help="input CSV file path containing continuation results (PPL, VERT, bitrate)",
    )
    parser.add_argument(
        "--x-axis",
        choices=["K", "bitrate"],
        default="bitrate",
        help="x-axis type",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output pdf path",
    )
    args = parser.parse_args()

    # load results
    df = pd.read_csv(args.continuation_result_csv, index_col=0)

    df = df[df["distance"].notna()]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(4.0, 3.0),
        constrained_layout=True,
        dpi=300,
    )
    fig.get_layout_engine().set(hspace=0.10)

    plot(axes, df, args.x_axis)

    output = args.output
    if args.output is None:
        output = f"./fig/continuation_score_{args.x_axis}.pdf"

    fig.savefig(output)
    print(f"Saved figure to {output}")
