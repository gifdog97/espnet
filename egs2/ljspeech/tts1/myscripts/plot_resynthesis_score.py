import argparse
import csv
import re
from pathlib import Path

try:
    import japanize_matplotlib  # noqa: F401
except ModuleNotFoundError:
    # Optional dependency (only for Japanese labels). Continue without it.
    pass
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


def parse_bitrate(file_path: str) -> dict[str, float]:
    """
    Input (tab-separated):
                N=20    N=40    ... N=280
        K=128   194.3   152.3   ...
        ...
    Returns:
        {"{N}-{K}": bitrate_value, ...}
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    Ns = [N_val.split("=")[1] for N_val in lines[0].strip().split("\t")]
    bitrate_dict: dict[str, float] = {}
    for line in lines[1:]:
        parts = line.strip().split("\t")
        K = parts[0].split("=")[1]
        values = list(map(float, parts[1:]))
        for N, value in zip(Ns, values):
            key = f"{N}-{K}"
            bitrate_dict[key] = value
    return bitrate_dict


def parse_error_rates(error_rates_csv: str) -> dict[str, dict[str, float]]:
    # {"tacotron2-20-128": {"wer": ..., "cer": ...}, ..., "gold": {"wer": ..., "cer": ...}}
    error_rates_dict = {}
    f = open(error_rates_csv, "r")
    reader = csv.DictReader(f)
    for row in reader:
        if row["Model"] == "Gold":
            setting = "gold"
        else:
            setting = f"{row['Model']}-{row['N']}-{row['K']}"
        error_rates_dict[setting] = {
            "wer": float(row["WER"]),
            "cer": float(row["CER"]),
        }
    f.close()
    return error_rates_dict


def parse_dsmetrics(dsmetrics_dir: str) -> dict[str, dict[str, float]]:
    # {"tacotron2-20-128": {"MCD": ...}, ..., "gold": {"MCD": None, ..., "utmos": value}}
    dsmetrics_dict = {}
    for csv_path in Path(dsmetrics_dir).glob("*.csv"):
        setting = csv_path.stem  # e.g., "tacotron2-20-128"
        df = pd.read_csv(csv_path)
        if "gold" in setting:
            dsmetrics_dict["gold"] = {
                "MCD": None,
                "Log_F0_RMSE": None,
                "speechBERTScore": None,
                "UTMOS": df["UTMOS"].iloc[0],
            }
        else:
            dsmetrics_dict[setting] = {
                "MCD": df["MCD"].iloc[0],
                "Log_F0_RMSE": df["LogF0RMSE"].iloc[0],
                "speechBERTScore": df["speechBERTScore"].iloc[0],
                "UTMOS": df["UTMOS"].iloc[0],
            }
    return dsmetrics_dict


SUFFIX_RE = re.compile(r"(\d+-\d+)$")  # 例: "20-128" を末尾から取る


def load_results(
    bitrate_csv: str = "csv/bitrate.csv",
    error_rates_csv: str = "csv/resynthesis_error_rates.csv",
    dsmetrics_dir: str = "csv/resynthesis_dsmetrics",
) -> pd.DataFrame:
    bitrate_dict = parse_bitrate(bitrate_csv)
    error_rates_dict = parse_error_rates(error_rates_csv)
    dsmetrics_dict = parse_dsmetrics(dsmetrics_dir)

    if set(error_rates_dict) != set(dsmetrics_dict):
        missing_in_ds = set(error_rates_dict) - set(dsmetrics_dict)
        missing_in_er = set(dsmetrics_dict) - set(error_rates_dict)
        raise ValueError(
            f"Key mismatch: missing_in_ds={missing_in_ds}, missing_in_er={missing_in_er}"
        )

    rows = {}
    for setting in error_rates_dict:
        row = {**error_rates_dict[setting], **dsmetrics_dict[setting]}
        # bitrate 付与
        if setting == "gold":
            row["bitrate"] = None  # or float("nan")
        else:
            m = SUFFIX_RE.search(setting)
            if not m:
                raise ValueError(
                    f"Cannot parse suffix (e.g., '20-128') from setting: {setting}"
                )
            suffix = m.group(1)
            row["bitrate"] = bitrate_dict.get(suffix)  # getにしておくと落ちにくい
            if row["bitrate"] is None:
                raise ValueError(
                    f"Bitrate not found for suffix={suffix} (setting={setting})"
                )
        rows[setting] = row
    return pd.DataFrame.from_dict(rows, orient="index")


_MARKERS = "osDv^<>X"
_N_LIST = [20, 40, 80, 120, 160, 200, 240, 280]
_I_LIST = list(range(7, 7 + 8))


def _x_values(df: pd.DataFrame, indices: list[str], x_axis: str) -> list[float]:
    if x_axis == "bitrate":
        return list(df.loc[indices, "bitrate"])
    if x_axis == "K":
        # Keep the original K-version behavior: use 1..8 and label ticks as 2^7..2^14.
        return list(range(1, len(indices) + 1))
    raise ValueError(f"Unknown x_axis: {x_axis}")


def plot(df: pd.DataFrame, x_axis: str) -> None:
    # prepare for plotting
    nrows = 5
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=(4.0, 6.6),
        dpi=300,
    )
    fig.subplots_adjust(
        left=0.12, right=0.98, top=0.96, bottom=0.07, hspace=0.16, wspace=0.08
    )

    ax_id = 0
    # --- Row 1: WER ---
    for num in range(2):
        ax = axes[ax_id][num]
        model_name = "tacotron2" if num == 0 else "vits"
        ax.set_title(model_name, fontsize=12)
        if x_axis == "bitrate":
            for N, marker in zip(_N_LIST, _MARKERS):
                indices = [f"{model_name}-{N}-{2**i}" for i in _I_LIST]
                ax.plot(
                    _x_values(df, indices, x_axis),
                    list(df.loc[indices, "wer"]),
                    marker=marker,
                    markersize=3,
                    linewidth=2,
                    alpha=0.7,
                )
            ax.set_xlim(0, 600)
            ax.set_xticks([0, 100, 200, 300, 400, 500, 600], minor=False)
            ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6], minor=False)
        else:  # x_axis == "K"
            for N in _N_LIST:
                indices = [f"{model_name}-{N}-{2**i}" for i in _I_LIST]
                ax.plot(
                    _x_values(df, indices, x_axis),
                    list(df.loc[indices, "wer"]),
                    marker=marker,
                    markersize=2.5,
                    linewidth=2,
                    alpha=0.7,
                )
            ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8], minor=False)

        ax.set_xticklabels([])  # hide x tick labels on the top plot
        ax.set_ylim(-5, 95)
        ax.set_yticks([0, 20, 40, 60, 80], minor=False)
        if num == 0:
            ax.set_yticklabels([0, 20, 40, 60, 80], fontsize=9)
        if num == 1:
            ax.set_yticklabels([])  # hide right y tick labels
        ax.grid()

    # --- Row 2: WER (<5) ---
    ax_id += 1
    for num in range(2):
        model_name = "tacotron2" if num == 0 else "vits"
        ax = axes[ax_id][num]
        if x_axis == "bitrate":
            for N, marker in zip(_N_LIST, _MARKERS):
                indices = [f"{model_name}-{N}-{2**i}" for i in _I_LIST]
                bitrate_values = list(df.loc[indices, "bitrate"])
                wer_values = list(df.loc[indices, "wer"])
                x, y = [], []
                for bitrate, wer in zip(bitrate_values, wer_values):
                    if wer < 5:
                        x.append(bitrate)
                        y.append(wer)
                ax.plot(x, y, marker=marker, markersize=3, linewidth=2, alpha=0.7)
            ax.set_xlim(0, 600)
            ax.set_xticks([0, 100, 200, 300, 400, 500, 600], minor=False)
            ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6], minor=False)
        else:
            for N in _N_LIST:
                indices = [f"{model_name}-{N}-{2**i}" for i in _I_LIST]
                wer_values = list(df.loc[indices, "wer"])
                x, y = [], []
                for i, wer in enumerate(wer_values):
                    if wer < 5:
                        x.append(i + 1)
                        y.append(wer)
                ax.plot(x, y, marker=marker, markersize=2.5, linewidth=2, alpha=0.7)
            ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8], minor=False)

        gold_wer = df.loc["gold", "wer"]
        ax.axhline(gold_wer, color="black", linestyle="--", label="Gold")
        ax.set_xticklabels([])  # hide x tick labels on the middle plot
        ax.set_ylim(0.5, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        if num == 0:
            ax.set_yticklabels([1, 2, 3, 4, 5], fontsize=9)
        if num == 1:
            ax.set_yticklabels([])  # hide right y tick labels
        ax.grid()

    # --- Row 3: speechBERTScore ---
    # ax_id += 1
    # for num in range(2):
    #     ax = axes[ax_id][num]
    #     model_name = "tacotron2" if num == 0 else "vits"
    #     if x_axis == "bitrate":
    #         for N, marker in zip(_N_LIST, _MARKERS):
    #             indices = [f"{model_name}-{N}-{2**i}" for i in _I_LIST]
    #             ax.plot(
    #                 _x_values(df, indices, x_axis),
    #                 list(df.loc[indices, "speechBERTScore"]),
    #                 marker=marker,
    #                 markersize=3,
    #                 linewidth=2,
    #                 alpha=0.7,
    #             )
    #         ax.set_xlim(0, 600)
    #         ax.set_xticks([0, 100, 200, 300, 400, 500, 600], minor=False)
    #         ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6], minor=False)
    #     else:  # x_axis == "K"
    #         for N in _N_LIST:
    #             indices = [f"{model_name}-{N}-{2**i}" for i in _I_LIST]
    #             ax.plot(
    #                 _x_values(df, indices, x_axis),
    #                 list(df.loc[indices, "speechBERTScore"]),
    #                 marker=marker,
    #                 markersize=2.5,
    #                 linewidth=2,
    #                 alpha=0.7,
    #             )
    #         ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8], minor=False)

    #     ax.set_xticklabels([])  # hide x tick labels on the top plot
    #     if num == 1:
    #         ax.set_yticklabels([])  # hide right y tick labels
    #     ax.set_ylim(-0.05, 1)
    #     ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1], minor=False)
    #     ax.grid()

    # --- Row 4: MCD ---
    ax_id += 1
    for num in range(2):
        ax = axes[ax_id][num]
        model_name = "tacotron2" if num == 0 else "vits"
        if x_axis == "bitrate":
            for N, marker in zip(_N_LIST, _MARKERS):
                indices = [f"{model_name}-{N}-{2**i}" for i in _I_LIST]
                ax.plot(
                    _x_values(df, indices, x_axis),
                    list(df.loc[indices, "MCD"]),
                    marker=marker,
                    markersize=3,
                    linewidth=2,
                    alpha=0.7,
                )
            ax.set_xlim(0, 600)
            ax.set_xticks([0, 100, 200, 300, 400, 500, 600], minor=False)
            ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6], minor=False)
        else:  # x_axis == "K"
            for N in _N_LIST:
                indices = [f"{model_name}-{N}-{2**i}" for i in _I_LIST]
                ax.plot(
                    _x_values(df, indices, x_axis),
                    list(df.loc[indices, "MCD"]),
                    marker=marker,
                    markersize=2.5,
                    linewidth=2,
                    alpha=0.7,
                )
            ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8], minor=False)

        ax.set_xticklabels([])  # hide x tick labels on the top plot
        ax.set_ylim(4.5, 10.5)
        ax.set_yticks([5, 6, 7, 8, 9, 10], minor=False)
        if num == 0:
            ax.set_yticklabels([5, 6, 7, 8, 9, 10], fontsize=9)
        if num == 1:
            ax.set_yticklabels([])  # hide right y tick labels
        ax.grid()

    # --- Row 5: Log_F0_RMSE ---
    ax_id += 1
    for num in range(2):
        ax = axes[ax_id][num]
        model_name = "tacotron2" if num == 0 else "vits"
        if x_axis == "bitrate":
            for N, marker in zip(_N_LIST, _MARKERS):
                indices = [f"{model_name}-{N}-{2**i}" for i in _I_LIST]
                ax.plot(
                    _x_values(df, indices, x_axis),
                    list(df.loc[indices, "Log_F0_RMSE"]),
                    marker=marker,
                    markersize=3,
                    linewidth=2,
                    alpha=0.7,
                )
            ax.set_xlim(0, 600)
            ax.set_xticks([0, 100, 200, 300, 400, 500, 600], minor=False)
            ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6], minor=False)
        else:  # x_axis == "K"
            for N in _N_LIST:
                indices = [f"{model_name}-{N}-{2**i}" for i in _I_LIST]
                ax.plot(
                    _x_values(df, indices, x_axis),
                    list(df.loc[indices, "Log_F0_RMSE"]),
                    marker=marker,
                    markersize=2.5,
                    linewidth=2,
                    alpha=0.7,
                )
            ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8], minor=False)

        ax.set_xticklabels([])  # hide x tick labels on the top plot
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1], minor=False)
        if num == 0:
            ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=9)
        if num == 1:
            ax.set_yticklabels([])  # hide right y tick labels
        ax.grid()

    # --- Row 6: UTMOS ---
    ax_id += 1
    for num in range(2):
        model_name = "tacotron2" if num == 0 else "vits"
        ax = axes[ax_id][num]
        gold_utmos = df.loc["gold", "UTMOS"]
        ax.axhline(gold_utmos, color="black", linestyle="--", label="Gold")
        if x_axis == "bitrate":
            for N, marker in zip(_N_LIST, _MARKERS):
                indices = [f"{model_name}-{N}-{2**i}" for i in _I_LIST]
                ax.plot(
                    _x_values(df, indices, x_axis),
                    list(df.loc[indices, "UTMOS"]),
                    label=f"N={N}",
                    marker=marker,
                    markersize=3,
                    linewidth=2,
                    alpha=0.7,
                )
            ax.set_xlim(0, 600)
            ax.set_xticks([0, 100, 200, 300, 400, 500, 600], minor=False)
            ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6], minor=False)
            ax.set_xlabel("Bitrate (x100) [bit/s]", fontsize=10)
        else:
            for N in _N_LIST:
                indices = [f"{model_name}-{N}-{2**i}" for i in _I_LIST]
                ax.plot(
                    _x_values(df, indices, x_axis),
                    list(df.loc[indices, "UTMOS"]),
                    marker=marker,
                    markersize=2.5,
                    linewidth=2,
                    alpha=0.7,
                )
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
            ax.set_xlabel("Cluster size (K)", fontsize=12)
        if num == 0:
            bbox = ax.get_position()
            hans, labs = ax.get_legend_handles_labels()
            fig.legend(
                handles=hans,
                labels=labs,
                loc="lower left",
                bbox_to_anchor=(bbox.x0, bbox.y0),
                fontsize=8,
                ncol=3,
            )
        ax.set_ylim(1, 5)
        ax.set_yticks([1, 2, 3, 4, 5], minor=False)
        if num == 0:
            ax.set_yticklabels([1, 2, 3, 4, 5], fontsize=9)
        if num == 1:
            ax.set_yticklabels([])  # hide right y tick labels
        ax.grid()

    row_labels = [
        r"WER[%]$\downarrow$",
        r"WER[%] (<5)",
        r"MCD[dB]$\downarrow$",
        r"LogF0 RMSE[cent]$\downarrow$",  # ←後述
        r"UTMOS$\uparrow$",
    ]

    for r, lab in enumerate(row_labels):
        axL = axes[r, 0]
        bb = axL.get_position()  # figure座標
        y = (bb.y0 + bb.y1) / 2
        fig.text(
            bb.x0 - 0.09,
            y,
            lab,  # ← x を固定できるのでガタつかない
            va="center",
            ha="center",
            rotation=90,
            fontsize=10,
        )

    output = f"./fig/resynthesis_score_{x_axis}.pdf"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot resynthesis scores (CER / UTMOS)."
    )
    parser.add_argument(
        "--x-axis",
        choices=["bitrate", "K"],
        default="bitrate",
        help="Choose x-axis. 'bitrate' or 'K'.",
    )
    parser.add_argument(
        "--bitrate-csv",
        type=str,
        default="csv/bitrate.csv",
        help="Path to bitrate.csv (tab-separated).",
    )
    parser.add_argument(
        "--error-rates-csv",
        type=str,
        default="csv/resynthesis_error_rates.csv",
        help="Path to resynthesis_error_rates.csv (tab-separated).",
    )
    parser.add_argument(
        "--dsmetrics-dir",
        type=str,
        default="csv/resynthesis_dsmetrics",
        help="Directory where resynthesis_dsmetrics are stored.",
    )
    args = parser.parse_args()

    # load results of tacotron2 and vits
    df = load_results(
        bitrate_csv=args.bitrate_csv,
        error_rates_csv=args.error_rates_csv,
        dsmetrics_dir=args.dsmetrics_dir,
    )
    df.to_csv("csv/resynthesis_result.csv", index=True)
    plot(df, x_axis=args.x_axis)


if __name__ == "__main__":
    main()
