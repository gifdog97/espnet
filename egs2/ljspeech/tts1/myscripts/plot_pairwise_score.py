from collections import defaultdict
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from myutils import extract_llm_scores, parse_bitrate

PAIRWISE_DIR = Path("pairwise")
_MARKERS = "osDv^<>X"

rcParams["pdf.fonttype"] = 42

# フォントファイルのパスを指定
font_path = (
    "/work/01/gk77/k77035/.local/share/fonts/Times New Roman/times new roman.ttf"
)

# フォントプロパティを作成
font_prop = fm.FontProperties(fname=font_path)

# グローバル設定に反映（全体に適用）
plt.rcParams["font.family"] = font_prop.get_name()


def plot_score(
    score_dict: dict[str, dict[str, list[float]]], bitrate_dict: dict[str, float]
):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(4.4, 2),
        dpi=300,
    )
    fig.subplots_adjust(
        left=0.12, right=0.98, top=0.88, bottom=0.22, hspace=0.16, wspace=0.08
    )
    for model, setting_scores in score_dict.items():
        if model == "tacotron2":
            ax = axes[0]
        else:
            ax = axes[1]
        n_to_points = defaultdict(lambda: defaultdict(list))
        for setting, scores in setting_scores.items():
            N, _ = setting.split("-")
            bitrate = bitrate_dict[setting]
            avg_score = sum(scores) / len(scores)
            sd = np.std(scores, ddof=1)
            sem = sd / np.sqrt(len(scores))
            ci95 = 1.96 * sem
            n_to_points[N]["bitrate"].append(bitrate)
            n_to_points[N]["avg_score"].append(avg_score)
            n_to_points[N]["ci95"].append(ci95)
        for i, N in enumerate(sorted(n_to_points.keys(), key=float)):
            points = n_to_points[N]
            triples = sorted(
                zip(points["bitrate"], points["avg_score"], points["ci95"]),
                key=lambda x: x[0],
            )  # x[0] = bitrate
            bitrate, avg_score, ci95 = (
                map(list, zip(*triples)) if triples else ([], [], [])
            )
            ax.errorbar(
                bitrate,
                avg_score,
                yerr=ci95,
                marker=_MARKERS[i],
                label=f"N={N}",
                markersize=3,
                linewidth=1.5,
            )
        ax.set_xlim(80, 520)
        xticks = [100, 200, 300, 400, 500]
        ax.set_xticks(xticks, minor=False)
        ax.set_xticklabels(np.array(xticks) // 100, fontsize=9)
        ax.set_xlabel("Bitrate (x100) [bps]")
        ax.set_ylim(-0.3, 0.2)
        yticks = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2]
        ax.set_yticks(yticks, minor=False)
        if model == "tacotron2":
            ax.set_yticklabels(yticks, fontsize=9)
            ax.set_ylabel("Average Score")
        else:
            ax.set_yticklabels([])  # hide right y tick labels
        ax.set_title(model)
        ax.legend(fontsize=8, ncol=2, columnspacing=0.8, loc="lower right")
        ax.grid()
    plt.savefig("fig/pairwise_score.pdf")
    print("Saved figure to fig/pairwise_score.pdf")


def main():
    bitrate_dict = parse_bitrate("csv/bitrate.csv")
    score_dict = defaultdict(lambda: defaultdict(list))
    for pairwise_dir in PAIRWISE_DIR.iterdir():
        if "_vs_" not in pairwise_dir.name:
            continue
        # tacotron vs. vits と vits vs. tacotron を無視
        # tacotron vs tacotron や vits vs vits だけ処理
        # if pairwise_dir.name.count("vits") == 1:
        #     continue
        # {model}-{N}-{K}-{temperature}_vs_{model}-{N}-{K}-{temperature}
        setting_X = pairwise_dir.name.split("_vs_")[0]
        model, N, K, _ = setting_X.split("-")
        score_dict[model][f"{N}-{K}"].extend(
            extract_llm_scores(pairwise_dir / "summary.txt")
        )
    plot_score(score_dict, bitrate_dict)


if __name__ == "__main__":
    main()
