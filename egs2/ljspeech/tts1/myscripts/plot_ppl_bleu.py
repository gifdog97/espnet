import itertools

import matplotlib.pyplot as plt
from numpy import sqrt
from tqdm import tqdm


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
    for model in ["tacotron2", "vits"]:
        print(model)
        NK_combinations = list(
            itertools.product(
                [20, 40, 80, 120, 160, 200, 240, 280], [2**i for i in range(7, 7 + 8)]
            )
        )
        for N, K in tqdm(NK_combinations):
            setting = f"{N}-{K}"
            plt.figure(figsize=(3, 2.5))
            gold_ppl_csv = "csv/ppl/gold.txt"
            gold_bleu_csv = "csv/bleu/gold.txt"
            gold_ppl = read_ppl(gold_ppl_csv)
            gold_bleu = read_bleu(gold_bleu_csv)
            plt.plot(gold_ppl, gold_bleu, "*", label="Gold Transcript")
            ppls = []
            bleus = []
            best_distance = float("inf")
            best_temperature, best_ppl, best_bleu = None, None, None
            temperatures = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
            for temperature in temperatures:
                if model == "tacotron2":
                    ppl_csv = f"csv/ppl/fixed_{setting}-{temperature}.txt"
                    bleu_csv = f"csv/bleu/fixed_{setting}-{temperature}.txt"
                elif model == "vits":
                    ppl_csv = f"csv/ppl/vits/fixed_{setting}-{temperature}.txt"
                    bleu_csv = f"csv/bleu/vits/fixed_{setting}-{temperature}.txt"
                ppl = read_ppl(ppl_csv)
                bleu = read_bleu(bleu_csv)
                ppls.append(ppl)
                bleus.append(bleu)
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
            plt.plot(ppls, bleus, ".", ls="-")
            plt.title(f"PPL vs VERT (N={N}, K={K})", fontsize=11)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.xlabel("PPL", fontsize=11)
            plt.ylabel("VERT", fontsize=11)
            plt.tight_layout()
            if model == "tacotron2":
                output_file = f"fig/continuation_plot/{setting}.pdf"
            elif model == "vits":
                output_file = f"fig/continuation_plot/vits/{setting}.pdf"
            plt.savefig(output_file, dpi=300)
            plt.close()
            if model == "tacotron2":
                score_tsv = f"csv/continuation_score/{setting}.tsv"
            elif model == "vits":
                score_tsv = f"csv/continuation_score/vits/{setting}.tsv"
            with open(score_tsv, "w") as f:
                f.write("temperature\tPPL\tVERT\tdistance\n")
                f.write(
                    f"{best_temperature}\t{best_ppl}\t{best_bleu}\t{format(best_distance, '.2e')}\n"
                )


if __name__ == "__main__":
    main()
