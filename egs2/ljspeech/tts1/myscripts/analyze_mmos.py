from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.patches import Rectangle
from scipy.stats import mannwhitneyu, t

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


# ----------------------------
# Multiple comparison corrections
# ----------------------------


def bonferroni_adjust(p):
    p = np.asarray(p, dtype=float)
    return np.clip(p * len(p), 0.0, 1.0)


def holm_adjust(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]

    adj_sorted = (m - np.arange(m)) * p_sorted
    adj_sorted = np.maximum.accumulate(adj_sorted)
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    adj = np.empty_like(adj_sorted)
    adj[order] = adj_sorted
    return adj


def fdr_bh_adjust(p):
    """Benjamini–Hochberg (FDR control)"""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]

    adj_sorted = p_sorted * m / (np.arange(1, m + 1))
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    adj = np.empty_like(adj_sorted)
    adj[order] = adj_sorted
    return adj


# ----------------------------
# Main function
# ----------------------------


def plot_mwu_heatmap(
    mos_dict,
    *,
    alpha=0.05,
    correction="holm",  # "holm", "bonferroni", "fdr_bh", "none"
    alternative="two-sided",
    figsize=(11, 9),
    value_fmt="{:+.2f}",
):
    # sort keys as N-K numeric
    settings = sorted(mos_dict.keys(), key=lambda s: tuple(map(int, s.split("-"))))
    n = len(settings)

    score_arrays = []
    for s in settings:
        scores = np.array([float(e["score"]) for e in mos_dict[s]], dtype=float)
        score_arrays.append(scores)

    means = np.array([arr.mean() for arr in score_arrays])
    mean_diff = means[:, None] - means[None, :]

    # compute upper triangle p-values
    pairs = []
    raw_p = []
    for i in range(n):
        for j in range(i + 1, n):
            res = mannwhitneyu(
                score_arrays[i],
                score_arrays[j],
                alternative=alternative,
                method="auto",
            )
            pairs.append((i, j))
            raw_p.append(res.pvalue)

    raw_p = np.array(raw_p)

    # apply correction
    if correction == "holm":
        adj_p = holm_adjust(raw_p)
    elif correction == "bonferroni":
        adj_p = bonferroni_adjust(raw_p)
    elif correction == "fdr_bh":
        adj_p = fdr_bh_adjust(raw_p)
    elif correction == "none":
        adj_p = raw_p.copy()
    else:
        raise ValueError("Unknown correction method")

    # build full matrix
    pvals_adj = np.ones((n, n))
    for (i, j), p in zip(pairs, adj_p):
        pvals_adj[i, j] = pvals_adj[j, i] = p

    signif = (pvals_adj < alpha) & (~np.eye(n, dtype=bool))

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mean_diff, aspect="auto")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(settings, rotation=90)
    ax.set_yticklabels(settings)

    ax.set_title(f"MWU mean diff (* if p_adj<{alpha}, correction={correction})")

    fig.colorbar(im, ax=ax, label="Mean difference")

    for i in range(n):
        for j in range(n):
            text = value_fmt.format(mean_diff[i, j])
            if signif[i, j]:
                text += "*"
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        linewidth=1.8,
                    )
                )
            ax.text(j, i, text, ha="center", va="center")

    fig.tight_layout()
    fig.savefig(f"fig/mwu_heatmap_{correction}.pdf", dpi=200)

    return {
        "settings": settings,
        "mean_diff": mean_diff,
        "pvals_adj": pvals_adj,
        "significant": signif,
        "correction": correction,
        "alpha": alpha,
    }


def extract_mmos_dict() -> dict[str, list[dict[str, int]]]:
    # mapping from setting (N-K) to list of scores
    mmos_dict = defaultdict(list)
    for csv_file in (MMOS_DIR / "summary").glob("*.csv"):
        N, K, _ = csv_file.name.replace(".csv", "").split("-")
        with open(csv_file) as f:
            if f"{N}-{K}" in mmos_dict:
                continue
            for line in f.readlines()[1:]:  # skip header
                # 5cae6a77b38ea60016e54889,LJ049-0061-0,3
                rater_id, sample_id, score = line.strip().split(",")
                mmos_dict[f"{N}-{K}"].append(
                    {"rater_id": rater_id, "sample_id": sample_id, "score": int(score)}
                )
    return mmos_dict


def plot_setting_ci(
    df_ci,
    mean_col="mean_mmos",
    lo_col="ci95_low",
    hi_col="ci95_high",
    label_col="setting",
    figsize=(6, 2.4),
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

    plt.ylim(2.1, 2.8)
    yticks = [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
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
            tcrit = t.ppf(1 - alpha / 2, df=n - 1)
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
    mmos_dict = extract_mmos_dict()

    df = per_setting_ci(mmos_dict, ci_method="t")
    df.to_csv("csv/mmos_per_setting_ci.csv", index=False)
    plot_setting_ci(df, outfile="fig/mmos_per_setting_ci.pdf")

    plot_mwu_heatmap(mmos_dict, correction="none", alpha=0.05)
    bonferroni_dict = plot_mwu_heatmap(mmos_dict, correction="bonferroni", alpha=0.05)
    bonferroni_df = pd.DataFrame(
        [
            {
                "setting_i": bonferroni_dict["settings"][i],
                "setting_j": bonferroni_dict["settings"][j],
                "mean_diff": bonferroni_dict["mean_diff"][i, j],
                "p_adj": bonferroni_dict["pvals_adj"][i, j],
                "significant": bonferroni_dict["significant"][i, j],
            }
            for i in range(len(bonferroni_dict["settings"]))
            for j in range(i + 1, len(bonferroni_dict["settings"]))
        ]
    )
    bonferroni_df.to_csv("csv/mmos_bonferroni.csv", index=False)
    plot_mwu_heatmap(mmos_dict, correction="holm", alpha=0.05)
    plot_mwu_heatmap(mmos_dict, correction="fdr_bh", alpha=0.05)


if __name__ == "__main__":
    main()
