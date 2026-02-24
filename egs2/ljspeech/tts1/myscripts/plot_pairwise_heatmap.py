from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from myutils import extract_llm_scores, parse_bitrate

PAIRWISE_DIR = Path("pairwise_10s")

rcParams["pdf.fonttype"] = 42
# フォントファイルのパスを指定
font_path = (
    "/work/01/gk77/k77035/.local/share/fonts/Times New Roman/times new roman.ttf"
)
mpl.rcParams["axes.unicode_minus"] = False


# フォントプロパティを作成
font_prop = fm.FontProperties(fname=font_path)

# グローバル設定に反映（全体に適用）
plt.rcParams["font.family"] = font_prop.get_name()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairwise_dir",
        type=Path,
        default=PAIRWISE_DIR,
        help="Directory containing pairwise LLM evaluation results",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default="fig/pairwise_10s_heatmap.pdf",
        help="Path to save the generated heatmap figure (PDF)",
    )
    return parser.parse_args()


def plot_pairwise(
    score_dict: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    figsize: tuple[float, float] = (6.8, 6.2),
    cmap: str = "bwr",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    save_path: str | None = None,
):
    # ---------- helpers ----------
    def _parse_setting(s: str) -> tuple[str, int, int]:
        model, N, K = s.split("-")
        return model, int(N), int(K)

    def _safe_mean(xs: list[float]) -> float:
        if xs is None or len(xs) == 0:
            return float("nan")
        return float(np.mean(xs))

    def _short_label(s: str) -> str:
        # "tacotron2-80-1024" -> "80-1024"
        model, N, K = s.split("-")
        return f"{N}-{K}"

    # ---------- collect all settings ----------
    all_settings = set(score_dict.keys())
    for row_k, cols in score_dict.items():
        all_settings.add(row_k)
        for col_k in cols.keys():
            all_settings.add(col_k)

    parsed = []
    for s in all_settings:
        model, N, K = _parse_setting(s)
        parsed.append((model, N, K, s))

    Ns = sorted({N for _, N, _, _ in parsed})
    Ks = sorted({K for _, _, K, _ in parsed})
    if not Ns or not Ks:
        raise ValueError("No settings found in score_dict.")

    def _ordered_settings_for_model(model_name: str) -> list[str]:
        existing = {(m, N, K): s for (m, N, K, s) in parsed if m == model_name}
        order: list[str] = []
        for N in Ns:
            for K in Ks:
                key = (model_name, N, K)
                if key in existing:
                    order.append(existing[key])
        return order

    taco_order = _ordered_settings_for_model("tacotron2")
    vits_order = _ordered_settings_for_model("vits")
    row_order = taco_order + vits_order
    col_order = list(row_order)

    taco_len = len(taco_order)
    vits_len = len(vits_order)
    n_total = len(row_order)

    # ---------- build matrix ----------
    M = np.full((n_total, n_total), np.nan, dtype=float)

    row_bitrate: dict[str, Any] = {}
    for r in row_order:
        if r in score_dict:
            for _, entry in score_dict[r].items():
                if isinstance(entry, dict) and "bitrate_X" in entry:
                    row_bitrate[r] = entry["bitrate_X"]
                    break
        if r not in row_bitrate:
            for rr, cols in score_dict.items():
                for cc, entry in cols.items():
                    if rr == r and isinstance(entry, dict) and "bitrate_X" in entry:
                        row_bitrate[r] = entry["bitrate_X"]
                        break
                if r in row_bitrate:
                    break

    for i, r in enumerate(row_order):
        for j, c in enumerate(col_order):
            entry = score_dict.get(r, {}).get(c, None)
            if entry is None:
                continue
            M[i, j] = _safe_mean(entry.get("scores", []))

    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # ===== tick labels: model名は消す =====
    row_labels = [f"{_short_label(r)} ({row_bitrate.get(r, '?')})" for r in row_order]
    col_labels = [_short_label(c) for c in col_order]  # (bitrate) も model も書かない

    ax.set_yticks(np.arange(n_total))
    ax.set_yticklabels(row_labels, fontsize=8)

    ax.set_xticks(np.arange(n_total))
    ax.set_xticklabels(col_labels, rotation=90, ha="center", va="top", fontsize=8)

    # ===== モデル境界線（自動） =====

    # N ごとの境界
    ax.axhline(-0.5 + 6, color="black", linewidth=2)
    ax.axhline(-0.5 + 6 + 5, color="black", linewidth=2)
    ax.axhline(-0.5 + 6 + 5 + 6, color="black", linewidth=2)
    # ax.axhline(-0.5 + 6 + 5 + 6 + 4, color="black", linewidth=2)
    ax.axhline(-0.5 + 6 + 5 + 6 + 4 + 7, color="black", linewidth=2)
    ax.axhline(-0.5 + 6 + 5 + 6 + 4 + 7 + 7, color="black", linewidth=2)
    ax.axvline(-0.5 + 6, color="black", linewidth=2)
    ax.axvline(-0.5 + 6 + 5, color="black", linewidth=2)
    ax.axvline(-0.5 + 6 + 5 + 6, color="black", linewidth=2)
    # ax.axvline(-0.5 + 6 + 5 + 6 + 4, color="black", linewidth=2)
    ax.axvline(-0.5 + 6 + 5 + 6 + 4 + 7, color="black", linewidth=2)
    ax.axvline(-0.5 + 6 + 5 + 6 + 4 + 7 + 7, color="black", linewidth=2)

    # tacotron2 と vits の境界
    ax.axhline(taco_len - 0.5, color="green", linewidth=4)
    ax.axvline(taco_len - 0.5, color="green", linewidth=4)

    # ===== 上部と右部に bracket/矢印注釈 =====
    # annotate は Axes座標で置く（x,y が 0..1）
    def _add_top_bracket(x0: float, x1: float, text: str, y: float = 1.01):
        ax.annotate(
            "",
            xy=(x1, y),
            xytext=(x0, y),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="<->", lw=1.8, color="black"),
            annotation_clip=False,
        )
        ax.text(
            (x0 + x1) / 2,
            y + 0.01,
            text,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
            clip_on=False,
        )

    def _add_right_bracket(y0: float, y1: float, text: str, x: float = 1.01):
        ax.annotate(
            "",
            xy=(x, y1),
            xytext=(x, y0),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="<->", lw=1.8, color="black"),
            annotation_clip=False,
        )
        ax.text(
            x + 0.01,
            (y0 + y1) / 2,
            text,
            transform=ax.transAxes,
            ha="left",
            va="center",
            rotation=90,
            fontsize=10,
            clip_on=False,
        )

    # セル index -> Axes比率に変換
    def _frac(i: int) -> float:
        # 0..n_total を 0..1 に
        return i / n_total

    # tacotron2 範囲: [0, taco_len)
    if taco_len > 0:
        _add_top_bracket(_frac(0), _frac(taco_len), "tacotron2")
        # imshow は y=0 が上なので、Axes座標の y も上から下に増える
        _add_right_bracket(
            _frac(n_total) - _frac(taco_len), _frac(n_total), "tacotron2"
        )

    # vits 範囲: [taco_len, n_total)
    if vits_len > 0:
        _add_top_bracket(_frac(taco_len), _frac(n_total), "vits")
        _add_right_bracket(_frac(0), _frac(n_total) - _frac(taco_len), "vits")

    if title is not None:
        ax.set_title(title)

    divider = make_axes_locatable(ax)

    # size が「横幅」だけを決める
    cax = divider.append_axes("right", size="2.5%", pad=0.3)

    cbar = fig.colorbar(im, ax=ax, cax=cax)
    ticks = np.arange(-0.4, 0.4 + 1e-9, 0.1)
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=8)

    # bracket を外側に出すので余白を少し増やす
    fig.tight_layout()
    fig.subplots_adjust(top=0.90, right=0.90)

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax


def main():
    args = parse_args()
    bitrate_dict = parse_bitrate("csv/bitrate.csv")
    score_dict = defaultdict(lambda: defaultdict(dict))
    for pairwise_dir in args.pairwise_dir.iterdir():
        if "_vs_" not in pairwise_dir.name:
            continue
        # {model}-{N}-{K}-{temperature}_vs_{model}-{N}-{K}-{temperature}
        setting_X, setting_Y = pairwise_dir.name.split("_vs_")
        model_X, N_X, K_X, _ = setting_X.split("-")
        bitrate_X = bitrate_dict[f"{N_X}-{K_X}"]
        model_Y, N_Y, K_Y, _ = setting_Y.split("-")
        bitrate_Y = bitrate_dict[f"{N_Y}-{K_Y}"]
        score_dict[f"{model_X}-{N_X}-{K_X}"][f"{model_Y}-{N_Y}-{K_Y}"] = {
            "bitrate_X": bitrate_X,
            "bitrate_Y": bitrate_Y,
            "scores": extract_llm_scores(pairwise_dir / "summary.txt"),
        }
    plot_pairwise(score_dict, save_path=args.output_path)
    print(f"Saved heatmap figure to {args.output_path}")


if __name__ == "__main__":
    main()
