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
                result[key] = round(float(value), 3)
    return result


def load_results():
    result_dict = {}
    for MS in [280, 240, 200, 160, 120, 80, 40, 20]:
        for i in range(7, 15):
            result = parse_result_file(
                f"../exp/fixed_{MS}-{2**i}_dedup/tts_train_raw_phn_none/decode_with_ljspeech_style_melgan.v1/dev/scoring/versa_eval/avg_result.txt"
            )
            result_dict[f"fixed_{MS}-{2**i}_dedup"] = {
                "wer": result["whisper_wer"],
                "cer": result["whisper_cer"],
                "utmos": result["utmos"],
            }
    return pd.DataFrame.from_dict(result_dict, orient="index")


df = load_results()

df.to_csv("csv/asr-utmos.csv", index=True)

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
    nrows=1, ncols=2, sharex=True, figsize=(5.6, 2.8), constrained_layout=True
)


def plot_heatmap(title, task_type, num):
    ax = axes[num]
    ax.set_title(title)
    for MS in [20, 40, 80, 120, 160, 200, 240, 280]:
        indices = [f"fixed_{MS}-{2**i}_dedup" for i in range(7, 7 + 8)]
        ax.plot(range(8), list(df.loc[indices, task_type]))
    # im = ax.imshow(data, vmin=0.45, vmax=0.75, aspect="equal")
    # if num == 0:
    #     ax.set_ylabel(r"segment ($N$)")
    #     ax.set_yticks([i for i in range(8)], minor=False)
    #     ax.set_yticklabels(y, minor=False, fontsize=8)
    # else:
    #     ax.set_yticks([])
    #     ax.yaxis.set_ticklabels([])
    ax.set_xlabel("clusters (K)")
    ax.set_xticks([i for i in range(8)], minor=False)
    ax.set_xticklabels(x, minor=False, fontsize=9)
    # for i in range(8):
    #     for j in range(8):
    #         val = f"{data[i][j]:.2f}".split(".")[1]
    #         ax.text(
    #             j,
    #             i,
    #             f".{val}",
    #             ha="center",
    #             va="center",
    #             color="w",
    #         )
    if num == 1:
        ax.legend(
            [f"N={MS}" for MS in [20, 40, 80, 120, 160, 200, 240, 280]],
            loc="lower left",
            bbox_to_anchor=(0, 0),
            fontsize=9,
        )


plot_heatmap("CER", "cer", 0)
plot_heatmap("UTMOS", "utmos", 1)

fig.savefig("./fig/fig-cer_utmos.pdf")
