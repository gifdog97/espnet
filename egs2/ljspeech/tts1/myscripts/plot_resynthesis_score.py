import argparse
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
        {"fixed_{N}-{K}_dedup": bitrate_value, ...}
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
            key = f"fixed_{N}-{K}_dedup"
            bitrate_dict[key] = value
    return bitrate_dict


def parse_result_file(file_path: str) -> dict[str, float]:
    """
    Parse the result file and return a dict.
    Format: key: value\n (or key value\n)
    """
    result: dict[str, float] = {}
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            key, value = line.strip().split(" ")
            if key.endswith(":"):
                key = key[:-1]
            result[key] = float(value)
    return result


def load_results(tts_name: str, bitrate_csv: str = "csv/bitrate.csv") -> pd.DataFrame:
    result_dict: dict[str, dict[str, float]] = {}
    bitrate_dict = parse_bitrate(bitrate_csv)

    for N in [280, 240, 200, 160, 120, 80, 40, 20]:
        for i in range(7, 7 + 8):
            exp_name = f"fixed_{N}-{2**i}_dedup"
            if tts_name == "tacotron2":
                result = parse_result_file(
                    f"../exp/{exp_name}/tts_train_raw_phn_none/"
                    "decode_with_ljspeech_style_melgan.v1/dev/scoring/versa_eval/avg_result.txt"
                )
            elif tts_name == "vits":
                result = parse_result_file(
                    f"../exp/{exp_name}-vits/tts_train_vits_raw_phn_none/"
                    "decode_with_vits/dev/scoring/versa_eval/avg_result.txt"
                )
            else:
                raise ValueError(f"Unknown tts_name: {tts_name}")

            result_dict[exp_name] = {
                "wer": round(100 * result["whisper_wer"], 3),
                "cer": round(100 * result["whisper_cer"], 3),
                "utmos": round(result["utmos"], 3),
                "bitrate": round(bitrate_dict[exp_name]),
            }
    return pd.DataFrame.from_dict(result_dict, orient="index")


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


def plot(axes, df: pd.DataFrame, num: int, x_axis: str) -> None:
    # --- Row 1: CER ---
    ax = axes[0][num]
    ax.grid()
    ax.set_title("Tacotron2" if num == 0 else "VITS", fontsize=12)

    if x_axis == "bitrate":
        for N, marker in zip(_N_LIST, _MARKERS):
            indices = [f"fixed_{N}-{2**i}_dedup" for i in _I_LIST]
            ax.plot(
                _x_values(df, indices, x_axis),
                list(df.loc[indices, "cer"]),
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
            indices = [f"fixed_{N}-{2**i}_dedup" for i in _I_LIST]
            ax.plot(
                _x_values(df, indices, x_axis),
                list(df.loc[indices, "cer"]),
                marker="o",
                markersize=2.5,
                linewidth=2,
                alpha=0.7,
            )
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8], minor=False)

    ax.set_xticklabels([])  # hide x tick labels on the top plot
    if num == 1:
        ax.set_yticklabels([])  # hide right y tick labels
    if num == 0:
        ax.set_ylabel(r"CER$\downarrow$", fontsize=12)
    ax.set_ylim(-5, 95)
    ax.set_yticks([0, 20, 40, 60, 80], minor=False)

    # --- Row 2: CER (<10) ---
    ax = axes[1][num]
    ax.grid()

    if x_axis == "bitrate":
        for N, marker in zip(_N_LIST, _MARKERS):
            indices = [f"fixed_{N}-{2**i}_dedup" for i in _I_LIST]
            bitrate_values = list(df.loc[indices, "bitrate"])
            cer_values = list(df.loc[indices, "cer"])
            x, y = [], []
            for bitrate, cer in zip(bitrate_values, cer_values):
                if cer < 10:
                    x.append(bitrate)
                    y.append(cer)
            ax.plot(x, y, marker=marker, markersize=3, linewidth=2, alpha=0.7)
        ax.set_xlim(0, 600)
        ax.set_xticks([0, 100, 200, 300, 400, 500, 600], minor=False)
        ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6], minor=False)
    else:
        for N in _N_LIST:
            indices = [f"fixed_{N}-{2**i}_dedup" for i in _I_LIST]
            cer_values = list(df.loc[indices, "cer"])
            x, y = [], []
            for i, cer in enumerate(cer_values):
                if cer < 10:
                    x.append(i + 1)
                    y.append(cer)
            ax.plot(x, y, marker="o", markersize=2.5, linewidth=2, alpha=0.7)
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8], minor=False)

    ax.set_xticklabels([])  # hide x tick labels on the middle plot
    if num == 0:
        ax.set_ylabel(r"CER (<10)", fontsize=12)
    if num == 1:
        ax.set_yticklabels([])  # hide right y tick labels
    ax.set_ylim(0, 10)
    ax.set_yticks([0, 2, 4, 6, 8, 10])

    # --- Row 3: UTMOS ---
    ax = axes[2][num]
    ax.grid()

    if x_axis == "bitrate":
        for N, marker in zip(_N_LIST, _MARKERS):
            indices = [f"fixed_{N}-{2**i}_dedup" for i in _I_LIST]
            ax.plot(
                _x_values(df, indices, x_axis),
                list(df.loc[indices, "utmos"]),
                marker=marker,
                markersize=3,
                linewidth=2,
                alpha=0.7,
            )
        ax.set_xlim(0, 600)
        ax.set_xticks([0, 100, 200, 300, 400, 500, 600], minor=False)
        ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6], minor=False)
        if num == 1:
            ax.set_yticklabels([])  # hide right y tick labels
            ax.legend(
                [f"N={N}" for N in _N_LIST],
                loc="lower right",
                bbox_to_anchor=(1, 0),
                fontsize=8,
                ncol=2,
            )
        ax.set_xlabel("Bitrate (x100)", fontsize=12)
    else:
        for N in _N_LIST:
            indices = [f"fixed_{N}-{2**i}_dedup" for i in _I_LIST]
            ax.plot(
                _x_values(df, indices, x_axis),
                list(df.loc[indices, "utmos"]),
                marker="o",
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
        if num == 0:
            ax.legend(
                [f"N={N}" for N in _N_LIST],
                loc="lower left",
                bbox_to_anchor=(0, 0),
                fontsize=8,
                ncol=2,
            )
        if num == 1:
            ax.set_yticklabels([])  # hide right y tick labels
        ax.set_xlabel("Cluster size (K)", fontsize=12)

    ax.set_ylim(2.25, 4.5)
    ax.set_yticks([2.5, 3.0, 3.5, 4.0, 4.5], minor=False)
    if num == 0:
        ax.set_ylabel(r"UTMOS$\uparrow$", fontsize=12)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot resynthesis scores (CER / UTMOS) for ICASSP figure."
    )
    parser.add_argument(
        "--x-axis",
        choices=["bitrate", "K"],
        default="bitrate",
        help="Choose x-axis. 'bitrate' reproduces plot_resynthesis_score_icassp.py, "
        "'K' reproduces plot_resynthesis_score_icassp_K.py.",
    )
    parser.add_argument(
        "--bitrate-csv",
        type=str,
        default="csv/bitrate.csv",
        help="Path to bitrate.csv (tab-separated).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the PDF. Default depends on --x-axis.",
    )
    args = parser.parse_args()

    # load results of tacotron2 and vits
    df_t = load_results("tacotron2", bitrate_csv=args.bitrate_csv)
    df_v = load_results("vits", bitrate_csv=args.bitrate_csv)
    df_t.to_csv("csv/resynthesis_result.csv", index=True)
    df_v.to_csv("csv/resynthesis_result-vits.csv", index=True)

    # prepare for plotting
    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(4.6, 4.6),
        constrained_layout=True,
        dpi=300,
    )
    fig.get_layout_engine().set(
        hspace=0.10
    )  # default is 0.20; larger => more row spacing

    plot(axes, df_t, 0, x_axis=args.x_axis)
    plot(axes, df_v, 1, x_axis=args.x_axis)

    output = args.output
    if output is None:
        output = f"./fig/resynthesis_score_{args.x_axis}.pdf"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
