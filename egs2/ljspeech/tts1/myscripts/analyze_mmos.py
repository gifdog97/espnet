from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from myutils import extract_llm_scores
from scipy import stats
from scipy.stats import spearmanr

PAIRWISE_DIR = Path("pairwise")
MMOS_DIR = Path("mmos_results")


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


def bootstrap_ci_mean(
    x: np.ndarray, B: int = 20000, seed: int = 0
) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI for the mean of x."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    n = x.size
    idx = rng.integers(0, n, size=(B, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def per_setting_ci(
    mmos_dict: Dict[str, List[Dict[str, Any]]],
    ci_method: str = "bootstrap",  # "bootstrap" or "t"
    alpha: float = 0.05,
    bootstrap_B: int = 20000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    For each setting:
      1) aggregate to sample-level mean over raters: mean(score | sample_id)
      2) compute setting mean over samples
      3) compute 95% CI for the setting mean

    Returns DataFrame with one row per setting.
    """
    rows_out = []
    for key, rows in mmos_dict.items():
        df = pd.DataFrame(rows)
        if "sample_id" not in df.columns or "score" not in df.columns:
            raise ValueError(
                f"mmos_dict[{key!r}] must contain dicts with keys 'sample_id' and 'score'."
            )

        df = df.copy()
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df = df.dropna(subset=["sample_id", "score"])

        # sample-level means (ideally 50 values)
        sample_means = (
            df.groupby("sample_id", sort=False)["score"].mean().to_numpy(dtype=float)
        )
        n_samples = int(sample_means.size)

        if n_samples < 2:
            mean_ = float(np.mean(sample_means)) if n_samples == 1 else float("nan")
            lo, hi = float("nan"), float("nan")
            se = float("nan")
        else:
            mean_ = float(sample_means.mean())
            se = float(sample_means.std(ddof=1) / np.sqrt(n_samples))

            if ci_method == "bootstrap":
                lo, hi = bootstrap_ci_mean(sample_means, B=bootstrap_B, seed=seed)
            elif ci_method == "t":
                # classic t CI: mean ± t_{1-α/2, n-1} * SE
                tcrit = stats.t.ppf(1 - alpha / 2, df=n_samples - 1)
                lo, hi = mean_ - tcrit * se, mean_ + tcrit * se
                lo, hi = float(lo), float(hi)
            else:
                raise ValueError("ci_method must be 'bootstrap' or 't'.")

        rows_out.append(
            dict(
                setting=key,
                n_unique_samples=n_samples,
                mean_mmos=mean_,
                se_over_samples=se,
                ci95_low=lo,
                ci95_high=hi,
            )
        )

    out = (
        pd.DataFrame(rows_out)
        .sort_values("mean_mmos", ascending=False)
        .reset_index(drop=True)
    )
    return out


def plot_setting_ci_rank(
    df_ci,
    figsize=(8, 10),
    xlabel="mMOS",
    title="mMOS with 95% CI (sorted)",
    outfile=None,
):
    """
    df_ci must contain columns:
      - setting
      - mean_mmos
      - ci95_low
      - ci95_high
    """

    # sort by mean descending
    df = df_ci.sort_values("mean_mmos", ascending=False).reset_index(drop=True)

    means = df["mean_mmos"].to_numpy()
    lo = df["ci95_low"].to_numpy()
    hi = df["ci95_high"].to_numpy()
    labels = df["setting"].astype(str).to_list()

    # asymmetric error bars
    err_low = means - lo
    err_high = hi - means
    yerr = np.vstack([err_low, err_high])

    y = np.arange(len(df))

    plt.figure(figsize=figsize)
    plt.errorbar(
        means,
        y,
        xerr=yerr,
        fmt="o",
        capsize=3,
    )

    plt.yticks(y, labels)
    plt.gca().invert_yaxis()  # top = best
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, dpi=200)
    else:
        plt.show()


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
    # df = per_setting_ci(mmos_dict, ci_method="bootstrap", bootstrap_B=20000, seed=0)
    df = per_setting_ci(mmos_dict, ci_method="t")
    df.to_csv("csv/mmos_per_setting_ci.csv", index=False)
    plot_setting_ci_rank(df, outfile="fig/mmos_per_setting_ci.pdf")


if __name__ == "__main__":
    main()
