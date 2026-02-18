from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from myutils import extract_llm_scores
from scipy import stats
from scipy.stats import spearmanr

PAIRWISE_DIR = Path("pairwise")
MMOS_DIR = Path("mmos_results")
rcParams["pdf.fonttype"] = 42

plt.rcParams["axes.unicode_minus"] = False

# フォントファイルのパスを指定
font_path = (
    "/work/01/gk77/k77035/.local/share/fonts/Times New Roman/times new roman.ttf"
)

# フォントプロパティを作成
font_prop = fm.FontProperties(fname=font_path)

# グローバル設定に反映（全体に適用）
plt.rcParams["font.family"] = font_prop.get_name()


def extract_result_dict():
    # extract llm_score_dict and mmos_dict
    # mapping from setting (N-K) to list of scores
    llm_score_dict = defaultdict(list)
    mmos_dict = defaultdict(list)
    for pairwise_dir in PAIRWISE_DIR.iterdir():
        if "_vs_" not in pairwise_dir.name:
            continue
        # tacotron vs tacotron だけ処理
        if pairwise_dir.name.count("tacotron2") != 2:
            continue
        setting_X = pairwise_dir.name.split("_vs_")[0]
        model, N, K, temperature = setting_X.split("-")
        llm_score_dict[f"{N}-{K}"].extend(
            extract_llm_scores(pairwise_dir / "summary.txt")
        )
        with open(MMOS_DIR / "summary" / f"{N}-{K}-{temperature}.csv") as f:
            if f"{N}-{K}" in mmos_dict:
                continue
            for line in f.readlines()[1:]:  # skip header
                # 5cae6a77b38ea60016e54889,LJ049-0061-0,3
                rater_id, sample_id, score = line.strip().split(",")
                mmos_dict[f"{N}-{K}"].append(
                    {"rater_id": rater_id, "sample_id": sample_id, "score": int(score)}
                )
    return llm_score_dict, mmos_dict


def plot_setting_ci(
    df_ci,
    mean_col="mean_mmos",
    lo_col="ci95_low",
    hi_col="ci95_high",
    label_col="setting",
    figsize=(6, 2.5),
    ylabel="MMOS",
    outfile=None,
):
    """
    x-axis: settings (sorted by mean desc)
    y-axis: mean MMOS with vertical 95% CI
    x tick labels rotated 90 degrees
    """

    # mean の高い順
    df = df_ci.sort_values(mean_col, ascending=False).reset_index(drop=True)

    means = df[mean_col].to_numpy(dtype=float)
    lo = df[lo_col].to_numpy(dtype=float)
    hi = df[hi_col].to_numpy(dtype=float)
    labels = df[label_col].astype(str).to_list()

    # asymmetric vertical error bars
    err_low = means - lo
    err_high = hi - means
    yerr = np.vstack([err_low, err_high])

    x = np.arange(len(df))

    plt.figure(figsize=figsize)
    plt.errorbar(
        x,
        means,
        yerr=yerr,
        fmt="o",
        capsize=3,
    )

    plt.ylim(2.3, 3.0)
    yticks = [2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
    plt.yticks(yticks, fontsize=10)
    plt.xticks(x, labels, rotation=90, fontsize=10)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, dpi=200)
    else:
        plt.show()


def per_setting_ci(
    mmos_dict: Dict[str, List[Dict[str, Any]]],
    ci_method: str = "t",  # "t" or "bootstrap"
    alpha: float = 0.05,
    bootstrap_B: int = 20000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Treat all raw scores within a setting as i.i.d. (e.g., 500 observations),
    and compute CI for the mean.

    Returns DataFrame with:
      - setting
      - n_scores
      - mean_mmos
      - se
      - ci95_low, ci95_high
    """
    rng = np.random.default_rng(seed)

    rows_out = []
    for key, rows in mmos_dict.items():
        df = pd.DataFrame(rows)
        if "score" not in df.columns:
            raise ValueError(f"mmos_dict[{key!r}] must contain dicts with key 'score'.")

        scores = (
            pd.to_numeric(df["score"], errors="coerce").dropna().to_numpy(dtype=float)
        )
        n = int(scores.size)
        if n < 2:
            rows_out.append(
                dict(
                    setting=key,
                    n_scores=n,
                    mean_mmos=float(np.mean(scores)) if n == 1 else np.nan,
                    se=np.nan,
                    ci95_low=np.nan,
                    ci95_high=np.nan,
                )
            )
            continue

        mean_ = float(scores.mean())
        se = float(scores.std(ddof=1) / np.sqrt(n))

        if ci_method == "t":
            tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
            lo, hi = mean_ - tcrit * se, mean_ + tcrit * se
        elif ci_method == "bootstrap":
            idx = rng.integers(0, n, size=(bootstrap_B, n))
            means = scores[idx].mean(axis=1)
            lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        else:
            raise ValueError("ci_method must be 't' or 'bootstrap'.")

        rows_out.append(
            dict(
                setting=key,
                n_scores=n,
                mean_mmos=mean_,
                se=se,
                ci95_low=float(lo),
                ci95_high=float(hi),
            )
        )

    out = (
        pd.DataFrame(rows_out)
        .sort_values("mean_mmos", ascending=False)
        .reset_index(drop=True)
    )
    return out


def main():
    llm_score_dict, mmos_dict = extract_result_dict()

    # plot scatter plot of averages_llm and averages_mmos
    averages_llm = [sum(scores) / len(scores) for scores in llm_score_dict.values()]
    averages_mmos = [
        sum(d["score"] for d in scores) / len(scores) for scores in mmos_dict.values()
    ]
    plt.scatter(averages_llm, averages_mmos)
    plt.xlabel("Average LLM Scores")
    plt.ylabel("Average MMOS Scores")
    plt.title("Scatter plot of Average LLM and MMOS Scores")
    plt.savefig("fig/llm_mmos_scatter_plot.pdf")
    plt.close()

    # calculate spearman correlation
    correlation, p_value = spearmanr(averages_llm, averages_mmos)
    print(f"Spearman correlation: {correlation}, p-value: {p_value}")
    df = per_setting_ci(mmos_dict, ci_method="t")
    df.to_csv("csv/mmos_per_setting_ci.csv", index=False)
    plot_setting_ci(df, outfile="fig/mmos_per_setting_ci.pdf")


if __name__ == "__main__":
    main()
