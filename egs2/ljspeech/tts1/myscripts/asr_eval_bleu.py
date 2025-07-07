# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Copied and modified from:
# https://github.com/facebookresearch/fairseq/blob/main/examples/textless_nlp/gslm/metrics/asr_metrics/self_auto_bleu.py

import warnings
from multiprocessing import Pool

import nltk
import numpy as np
from bleu_utils import sentence_bleu


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asr_transcript",
        type=str,
        help="Path to the transcript file.",
        default="transcription/fixed_20-128-0.3.txt",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to output scores.",
        default="csv/bleu/fixed_20-128-0.3.txt",
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    return args


def get_self_bleu(utterances, averaging_mode, weights):
    self_bleu = []

    for i in range(len(utterances)):
        hypo = utterances[i]
        rest = utterances[:i] + utterances[i + 1 :]

        self_bleu.append(
            sentence_bleu(
                rest,
                hypo,
                weights,
                no_length_penalty=True,
                averaging_mode=averaging_mode,
            )
        )

    return self_bleu


def get_self_bleu2_arithmetic(utterances):
    weights = (0.5, 0.5)  # equal weight for unigrams and bigrams
    return get_self_bleu(utterances, averaging_mode="arithmetic", weights=weights)


def get_self_bleu2_geometric(utterances):
    weights = (0.5, 0.5)
    return get_self_bleu(utterances, averaging_mode="geometric", weights=weights)


def get_auto_bleu2_arithmetic(utterances):
    weights = (0.5, 0.5)
    return [auto_bleu(u, mean_mode="arithmetic", weights=weights) for u in utterances]


def get_auto_bleu2_geometric(utterances):
    weights = (0.5, 0.5)
    return [auto_bleu(u, mean_mode="geometric", weights=weights) for u in utterances]


def get_auto_bleu3_geometric(utterances):
    weights = (1.0 / 3, 1.0 / 3, 1.0 / 3)
    return [auto_bleu(u, mean_mode="geometric", weights=weights) for u in utterances]


def get_auto_bleu3_arithmetic(utterances):
    weights = (1.0 / 3, 1.0 / 3, 1.0 / 3)
    return [auto_bleu(u, mean_mode="arithmetic", weights=weights) for u in utterances]


def get_self_bleu3_arithmetic(utterances):
    weights = (1.0 / 3, 1.0 / 3, 1.0 / 3)
    return get_self_bleu(utterances, averaging_mode="arithmetic", weights=weights)


def get_self_bleu3_geometric(utterances):
    weights = (1.0 / 3, 1.0 / 3, 1.0 / 3)
    return get_self_bleu(utterances, averaging_mode="geometric", weights=weights)


def auto_bleu(sentence, weights, mean_mode="arithmetic"):
    if len(sentence) <= 1:
        return 0

    N = len(weights)

    bleu_n = np.zeros([N])
    for n in range(N):
        targ_ngrams = list(nltk.ngrams(sentence, n + 1))
        for p in range(len(targ_ngrams)):
            left = sentence[:p]
            right = sentence[(p + n + 1) :]
            rest_ngrams = list(nltk.ngrams(left, n + 1)) + list(
                nltk.ngrams(right, n + 1)
            )
            # compute the nb of matching ngrams
            bleu_n[n] += targ_ngrams[p] in rest_ngrams
        bleu_n[n] /= len(targ_ngrams)  # average them to get a proportion

    weights = np.array(weights)
    if mean_mode == "arithmetic":
        return (bleu_n * weights).sum()
    elif mean_mode == "geometric":
        return (bleu_n**weights).prod()
    else:
        raise ValueError(f"Unknown agggregation mode {mean_mode}")


def main():
    args = get_args()

    with open(args.asr_transcript, "r") as fin:
        lines = fin.readlines()

    terms = [x.strip().split() for x in lines]

    if args.debug:
        terms = terms[:10]

    tasks = [
        ("Self-BLEU2-arithmetic", get_self_bleu2_arithmetic),
        # ("Self-BLEU2-geometric", get_self_bleu2_geometric),
        ("Auto-BLEU2-arithmetic", get_auto_bleu2_arithmetic),
        # ("Auto-BLEU2-geometric", get_auto_bleu2_geometric),
        # ("Self-BLEU3-arithmetic", get_self_bleu3_arithmetic),
        # ("Self-BLEU3-geometric", get_self_bleu3_geometric),
        # ("Auto-BLEU3-arithmetic", get_auto_bleu3_arithmetic),
        # ("Auto-BLEU3-geometric", get_auto_bleu3_geometric),
    ]

    n_processes = min(16, len(tasks))
    with Pool(n_processes) as pool:
        metrics = pool.map(run_f, [(t[1], terms) for t in tasks])

    with open(args.output_file, "w") as f:
        for (metric_name, _), metric in zip(tasks, metrics):
            metric, sem = np.mean(metric), np.std(metric) / np.sqrt(len(metric))
            metric, sem = [round(100 * x, 2) for x in [metric, sem]]
            f.write(f"{metric_name} {metric}+-{sem}\n")

        vert = np.sqrt(
            round(100 * np.mean(metrics[0]), 2) * round(100 * np.mean(metrics[1]), 2)
        )
        f.write(f"VERT {vert}")


def run_f(task_params):
    f, terms = task_params
    return f(terms)


if __name__ == "__main__":
    # NLTK produces warnings
    warnings.filterwarnings("ignore")

    main()
