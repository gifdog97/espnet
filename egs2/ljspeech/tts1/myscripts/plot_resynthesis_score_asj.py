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
    result = {}
    for line in lines[1:]:
        parts = line.strip().split("\t")
        K = parts[0].split("=")[1]
        values = list(map(float, parts[1:]))
        for N, value in zip(Ns, values):
            key = f"fixed_{N}-{K}_dedup"
            result[key] = value
    print(result)
    return result


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


def load_results():
    result_dict = {}
    bitrate_dict = parse_bitrate("csv/bitrate.csv")
    for MS in [280, 240, 200, 160, 120, 80, 40, 20]:
        for i in range(7, 7 + 8):
            result = parse_result_file(
                f"../exp/fixed_{MS}-{2**i}_dedup/tts_train_raw_phn_none/decode_with_ljspeech_style_melgan.v1/dev/scoring/versa_eval/avg_result.txt"
                # f"../exp/fixed_{MS}-{2**i}_dedup-vits/tts_train_vits_raw_phn_none/decode_with_vits/dev/scoring/versa_eval/avg_result.txt"
            )
            result_dict[f"fixed_{MS}-{2**i}_dedup"] = {
                "wer": round(100 * result["whisper_wer"], 3),
                "cer": round(100 * result["whisper_cer"], 3),
                "utmos": round(result["utmos"], 3),
                "bitrate": round(bitrate_dict[f"fixed_{MS}-{2**i}_dedup"]),
            }
    return pd.DataFrame.from_dict(result_dict, orient="index")


df = load_results()

df.to_csv("csv/asr-utmos.csv", index=True)
# df.to_csv("csv/asr-utmos-vits.csv", index=True)

x = [
    r"$2^{7}$",
    r"$2^{8}$",
    r"$2^{9}$",
    r"$2^{10}$",
    r"$2^{11}$",
    r"$2^{12}$",
    r"$2^{13}$",
    r"$2^{14}$",
]
y = [
    "280",
    "240",
    "200",
    "160",
    "120",
    "80",
    "40",
    "20",
]

fig, axes = plt.subplots(
    # nrows=2, ncols=1, sharex=True, figsize=(2.8, 4.0), constrained_layout=True, dpi=300
    nrows=3,
    ncols=1,
    sharex=True,
    figsize=(2.8, 5.6),
    constrained_layout=True,
    dpi=300,
)
fig.get_layout_engine().set(hspace=0.10)  # ← 0.20 が既定。大きいほど行間が広がる


def plot(title, task_type, num):
    ax = axes[num]
    ax.set_title(title, fontsize=12)
    for MS in [20, 40, 80, 120, 160, 200, 240, 280]:
        indices = [f"fixed_{MS}-{2**i}_dedup" for i in range(7, 7 + 8)]
        ax.plot(range(8), list(df.loc[indices, task_type]))
    if num == 0:
        yticks = [0, 20, 40, 60, 80]
    if num == 2:
        yticks = [2.5, 3.0, 3.5, 4.0]
    ax.set_yticks(yticks)
    yticklabels = ax.get_yticklabels()
    ax.set_yticklabels(yticklabels, fontsize=11)
    ax.set_xticks([i for i in range(8)], minor=False)
    if num == 2:
        ax.set_xlabel("clusters (K)", fontsize=10)
        ax.set_xticklabels(x, minor=False, fontsize=10)
        ax.legend(
            [f"N={MS}" for MS in [20, 40, 80, 120, 160, 200, 240, 280]],
            loc="lower left",
            bbox_to_anchor=(0, 0),
            fontsize=9,
            ncol=2,
        )


def plot_good(title, task_type, num):
    ax = axes[num]
    ax.set_title(title, fontsize=12)
    for MS in [20, 40, 80, 120, 160, 200, 240, 280]:
        indices = [f"fixed_{MS}-{2**i}_dedup" for i in range(7, 7 + 8)]
        cer_values = list(df.loc[indices, task_type])
        x, y = [], []
        for i, value in enumerate(cer_values):
            if value < 10:
                x.append(i)
                y.append(value)
        ax.plot(x, y)
    yticks = [0, 2, 4, 6, 8]
    ax.set_yticks(yticks)
    yticklabels = ax.get_yticklabels()
    ax.set_yticklabels(yticklabels, fontsize=11)
    ax.set_xticks([i for i in range(8)], minor=False)


plot(r"CER$\downarrow$", "cer", 0)
plot_good(r"CER (<10)", "cer", 1)
plot(r"UTMOS$\uparrow$", "utmos", 2)

fig.savefig("./fig/fig-cer_utmos.png")
# fig.savefig("./fig/fig-cer_utmos-vits.png")
