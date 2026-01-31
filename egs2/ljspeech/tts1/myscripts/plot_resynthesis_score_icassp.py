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


def parse_bitrate(file_path):
    """
    Input:
                N=20	N=40	N=80	N=120	N=160	N=200	N=240	N=280
        K=128	194.3	152.3	96.5	69.9	54.4	44.3	36.7	31.6
        K=256	237.2	181.0	116.1	81.9	63.2	50.8	42.6	36.8
        K=512	282.3	211.8	130.3	92.6	71.6	57.6	48.3	41.7
        K=1024	326.0	237.7	145.4	103.0	78.8	63.9	53.6	46.1
        K=2048	372.5	265.1	159.4	112.8	86.8	69.9	58.7	50.5
        K=4096	431.3	299.6	175.9	123.4	94.3	76.0	63.7	54.7
        K=8192	497.4	334.9	192.4	133.4	101.5	81.4	67.9	58.0
        K=16384	576.7	372.8	208.7	143.1	108.5	86.5	71.8	61.2
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    Ns = [N_val.split("=")[1] for N_val in lines[0].strip().split("\t")]
    bitrate_dict = {}
    for line in lines[1:]:
        parts = line.strip().split("\t")
        K = parts[0].split("=")[1]
        values = list(map(float, parts[1:]))
        for N, value in zip(Ns, values):
            key = f"fixed_{N}-{K}_dedup"
            bitrate_dict[key] = value
    return bitrate_dict


def parse_result_file(file_path):
    """
    Parse the result file and return a Dict.

    Args:
        file_path (str): The path to the result file.
        format: key: value\n
    """
    result = {}
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                key, value = line.strip().split(" ")
                if key.endswith(":"):
                    key = key[:-1]
                result[key] = float(value)
    return result


def load_results(tts_name: str):
    result_dict = {}
    bitrate_dict = parse_bitrate("csv/bitrate.csv")
    for N in [280, 240, 200, 160, 120, 80, 40, 20]:
        for i in range(7, 7 + 8):
            if tts_name == "tacotron2":
                result = parse_result_file(
                    f"../exp/fixed_{N}-{2**i}_dedup/tts_train_raw_phn_none/decode_with_ljspeech_style_melgan.v1/dev/scoring/versa_eval/avg_result.txt"
                )
            elif tts_name == "vits":
                result = parse_result_file(
                    f"../exp/fixed_{N}-{2**i}_dedup-vits/tts_train_vits_raw_phn_none/decode_with_vits/dev/scoring/versa_eval/avg_result.txt"
                )
            result_dict[f"fixed_{N}-{2**i}_dedup"] = {
                "wer": round(100 * result["whisper_wer"], 3),
                "cer": round(100 * result["whisper_cer"], 3),
                "utmos": round(result["utmos"], 3),
                "bitrate": round(bitrate_dict[f"fixed_{N}-{2**i}_dedup"]),
            }
    return pd.DataFrame.from_dict(result_dict, orient="index")


markers = "osDv^<>X"


def plot(axes, df, num):
    # plot CER
    ax = axes[0][num]
    ax.grid()
    if num == 0:
        # ax.set_title(r"CER$\downarrow$", fontsize=12)
        ax.set_title("Tacotron2", fontsize=12)
    elif num == 1:
        # ax.set_title(r"CER$\downarrow$", fontsize=12)
        ax.set_title("VITS", fontsize=12)
    for N, marker in zip([20, 40, 80, 120, 160, 200, 240, 280], markers):
        indices = [f"fixed_{N}-{2**i}_dedup" for i in range(7, 7 + 8)]
        ax.plot(
            list(df.loc[indices, "bitrate"]),
            list(df.loc[indices, "cer"]),
            marker=marker,
            markersize=3,
            linewidth=2,
            alpha=0.7,
        )
    ax.set_xlim(0, 600)
    ax.set_xticks([0, 100, 200, 300, 400, 500, 600], minor=False)
    ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6], minor=False)
    ax.set_xticklabels([])  # 上のグラフのx軸ラベルを消す
    if num == 1:
        ax.set_yticklabels([])  # 右のグラフのy軸ラベルを消す
    if num == 0:
        ax.set_ylabel(r"CER$\downarrow$", fontsize=12)
    ax.set_ylim(-5, 95)
    ax.set_yticks([0, 20, 40, 60, 80], minor=False)

    # plot CER (<10)
    ax = axes[1][num]
    ax.grid()
    for N, marker in zip([20, 40, 80, 120, 160, 200, 240, 280], markers):
        indices = [f"fixed_{N}-{2**i}_dedup" for i in range(7, 7 + 8)]
        bitrate_values = list(df.loc[indices, "bitrate"])
        cer_values = list(df.loc[indices, "cer"])
        x, y = [], []
        for bitrate, cer in zip(bitrate_values, cer_values):
            if cer < 10:
                x.append(bitrate)
                y.append(cer)
        ax.plot(x, y, marker=marker, markersize=3, linewidth=2, alpha=0.7)
    if num == 0:
        ax.set_ylabel(r"CER (<10)", fontsize=12)
    if num == 1:
        ax.set_yticklabels([])  # 右のグラフのy軸ラベルを消す
    ax.set_xlim(0, 600)
    ax.set_xticks([0, 100, 200, 300, 400, 500, 600], minor=False)
    ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6], minor=False)
    ax.set_xticklabels([])  # 上のグラフのx軸ラベルを消す
    ax.set_ylim(0, 10)
    ax.set_yticks([0, 2, 4, 6, 8, 10])

    # plot UTMOS
    ax = axes[2][num]
    ax.grid()
    for N, marker in zip([20, 40, 80, 120, 160, 200, 240, 280], markers):
        indices = [f"fixed_{N}-{2**i}_dedup" for i in range(7, 7 + 8)]
        ax.plot(
            list(df.loc[indices, "bitrate"]),
            list(df.loc[indices, "utmos"]),
            marker=marker,
            markersize=3,
            linewidth=2,
            alpha=0.7,
        )
    ax.set_ylim(2.25, 4.5)
    ax.set_yticks([2.5, 3.0, 3.5, 4.0, 4.5], minor=False)
    ax.set_xlim(0, 600)
    ax.set_xticks([0, 100, 200, 300, 400, 500, 600], minor=False)
    ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6], minor=False)
    if num == 0:
        ax.set_ylabel(r"UTMOS$\uparrow$", fontsize=12)
    if num == 1:
        ax.set_yticklabels([])  # 右のグラフのy軸ラベルを消す
        ax.legend(
            [f"N={N}" for N in [20, 40, 80, 120, 160, 200, 240, 280]],
            loc="lower right",
            bbox_to_anchor=(1, 0),
            fontsize=8,
            ncol=2,
        )
    ax.set_xlabel("Bitrate (x100)", fontsize=12)


if __name__ == "__main__":
    # load results of tacotron2 and vits
    df_t = load_results("tacotron2")
    df_v = load_results("vits")
    df_t.to_csv("csv/asr-utmos.csv", index=True)
    df_v.to_csv("csv/asr-utmos-vits.csv", index=True)

    # prepare for plotting
    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(4.6, 4.6),
        constrained_layout=True,
        dpi=300,
    )
    fig.get_layout_engine().set(hspace=0.10)  # ← 0.20 が既定。大きいほど行間が広がる

    plot(axes, df_t, 0)
    plot(axes, df_v, 1)

    fig.savefig("./fig/resynthesis_score.pdf")
