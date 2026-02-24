import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
from numpy import sqrt
from tqdm import tqdm


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bleu_dir",
        type=str,
        default="csv/bleu_10s",
        help="Directory containing BLEU score files for different settings and temperatures",
    )
    parser.add_argument(
        "--ppl_dir",
        type=str,
        default="csv/ppl_10s",
        help="Directory containing PPL score files for different settings and temperatures",
    )
    parser.add_argument(
        "--score_dir",
        type=str,
        default="csv/continuation_score_10s",
        help="Directory containing continuation score files for different settings",
    )
    parser.add_argument(
        "--figure_path",
        type=Path,
        default=Path("fig/continuation_plot_10s"),
        help="Directory to save the generated score figures (PDF)",
    )
    return parser.parse_args()


def read_ppl(file_path: str) -> float:
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("PPL_corpus"):
                _, ppl = line.strip().split("|")
                return float(ppl)


def read_bleu(file_path: str) -> float:
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("VERT"):
                _, bleu = line.strip().split(" ")
                return float(bleu)


def main():
    gold_ppl_csv = "csv/ppl/gold.txt"
    gold_bleu_csv = "csv/bleu/gold.txt"
    gold_ppl = read_ppl(gold_ppl_csv)
    gold_bleu = read_bleu(gold_bleu_csv)
    args = parse_args()
    for model in ["tacotron2", "vits"]:
        NK_combinations = list(
            itertools.product(
                [20, 40, 80, 120, 160, 200, 240, 280], [2**i for i in range(7, 7 + 8)]
            )
        )
        for N, K in tqdm(NK_combinations):
            setting = f"{N}-{K}"
            plt.figure(figsize=(3, 2.5))
            plt.plot(gold_ppl, gold_bleu, "*", label="Gold Transcript")
            ppls = []
            bleus = []
            best_distance = float("inf")
            best_temperature, best_ppl, best_bleu = None, None, None
            temperatures = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
            # skip invalid settings
            if not Path(f"{args.bleu_dir}/{model}/{setting}-0.3.txt").exists():
                continue
            for temperature in temperatures:
                ppl_csv = f"{args.ppl_dir}/{model}/{setting}-{temperature}.txt"
                bleu_csv = f"{args.bleu_dir}/{model}/{setting}-{temperature}.txt"
                ppl = read_ppl(ppl_csv)
                bleu = read_bleu(bleu_csv)
                ppls.append(ppl)
                bleus.append(bleu)
            # plot results
            plt.plot(ppls, bleus, ".", ls="-")
            plt.title(f"PPL vs VERT (N={N}, K={K})", fontsize=11)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.xlabel("PPL", fontsize=11)
            plt.ylabel("VERT", fontsize=11)
            plt.tight_layout()
            output_file = f"{args.figure_path}/{model}/{setting}.pdf"
            plt.savefig(output_file, dpi=300)
            plt.close()
            # Choose best temperature and save score
            min_ppl, max_ppl = min(ppls + [gold_ppl]), max(ppls + [gold_ppl])
            min_bleu, max_bleu = min(bleus + [gold_bleu]), max(bleus + [gold_bleu])
            ppls_normalized = [(ppl - min_ppl) / (max_ppl - min_ppl) for ppl in ppls]
            bleus_normalized = [
                (bleu - min_bleu) / (max_bleu - min_bleu) for bleu in bleus
            ]
            gold_ppl_normalized = (gold_ppl - min_ppl) / (max_ppl - min_ppl)
            gold_bleu_normalized = (gold_bleu - min_bleu) / (max_bleu - min_bleu)
            for temperature, ppl, bleu, ppl_normalized, bleus_normalized in zip(
                temperatures, ppls, bleus, ppls_normalized, bleus_normalized
            ):
                distance = sqrt(
                    (ppl_normalized - gold_ppl_normalized) ** 2
                    + (bleus_normalized - gold_bleu_normalized) ** 2
                )
                if distance < best_distance:
                    best_distance = distance
                    best_temperature, best_ppl, best_bleu = temperature, ppl, bleu
            score_tsv = f"{args.score_dir}/{model}/{setting}.tsv"
            with open(score_tsv, "w") as f:
                f.write("temperature\tPPL\tVERT\tdistance\n")
                f.write(
                    f"{best_temperature}\t{best_ppl}\t{best_bleu}\t{format(best_distance, '.2e')}\n"
                )


if __name__ == "__main__":
    main()
