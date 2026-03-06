# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Copied and modified from:
# https://github.com/facebookresearch/fairseq/blob/main/examples/textless_nlp/gslm/metrics/asr_metrics/self_auto_bleu.py

import warnings
from collections import defaultdict
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
        utt_id = utterances[i][0]
        hypo = utterances[i][1]
        rest = [u[1] for u in utterances[:i]] + [u[1] for u in utterances[i + 1 :]]

        self_bleu.append(
            (
                utt_id,
                sentence_bleu(
                    rest,
                    hypo,
                    weights,
                    no_length_penalty=True,
                    averaging_mode=averaging_mode,
                ),
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
    if len(sentence[1]) <= 1:
        return 0
    utt_id = sentence[0]

    N = len(weights)

    bleu_n = np.zeros([N])
    for n in range(N):
        targ_ngrams = list(nltk.ngrams(sentence[1], n + 1))
        for p in range(len(targ_ngrams)):
            left = sentence[1][:p]
            right = sentence[1][(p + n + 1) :]
            rest_ngrams = list(nltk.ngrams(left, n + 1)) + list(
                nltk.ngrams(right, n + 1)
            )
            # compute the nb of matching ngrams
            bleu_n[n] += targ_ngrams[p] in rest_ngrams
        bleu_n[n] /= len(targ_ngrams)  # average them to get a proportion

    weights = np.array(weights)
    if mean_mode == "arithmetic":
        return (utt_id, (bleu_n * weights).sum())
    elif mean_mode == "geometric":
        return (utt_id, (bleu_n**weights).prod())
    else:
        raise ValueError(f"Unknown agggregation mode {mean_mode}")


def all_scores(asr_transcript):
    with open(asr_transcript, "r") as fin:
        lines = fin.readlines()

    terms = [x.strip().split("|") for x in lines]

    tasks = [
        # ("Self-BLEU2-arithmetic", get_self_bleu2_arithmetic),
        ("Self-BLEU2-geometric", get_self_bleu2_geometric),
        # ("Auto-BLEU2-arithmetic", get_auto_bleu2_arithmetic),
        ("Auto-BLEU2-geometric", get_auto_bleu2_geometric),
        # ("Self-BLEU3-arithmetic", get_self_bleu3_arithmetic),
        # ("Self-BLEU3-geometric", get_self_bleu3_geometric),
        # ("Auto-BLEU3-arithmetic", get_auto_bleu3_arithmetic),
        # ("Auto-BLEU3-geometric", get_auto_bleu3_geometric),
    ]

    n_processes = min(16, len(tasks))
    with Pool(n_processes) as pool:
        id_and_metrics = pool.map(run_f, [(t[1], terms) for t in tasks])

    metric_dict = defaultdict(list)
    for (metric_name, _), id_and_metric in zip(tasks, id_and_metrics):
        for wav_id, m in id_and_metric:
            metric_dict[metric_name].append((wav_id, m))
        # metric, sem = np.mean(metric), np.std(metric) / np.sqrt(len(metric))
        # metric, sem = [round(100 * x, 2) for x in [metric, sem]]

    for m1, m2 in zip(metric_dict[tasks[0][0]], metric_dict[tasks[1][0]]):
        metric_dict["VERT"].append(
            (m1[0], round(np.sqrt(100 * m1[1] * 100 * m2[1]), 2))
        )
    # vert = round(np.sqrt(100 * np.mean(metrics[0]) * 100 * np.mean(metrics[1])), 2)

    return metric_dict


def main():
    args = get_args()

    with open(args.asr_transcript, "r") as fin:
        lines = fin.readlines()

    terms = [x.strip().split("|") for x in lines]

    if args.debug:
        terms = terms[:10]

    tasks = [
        # ("Self-BLEU2-arithmetic", get_self_bleu2_arithmetic),
        ("Self-BLEU2-geometric", get_self_bleu2_geometric),
        # ("Auto-BLEU2-arithmetic", get_auto_bleu2_arithmetic),
        ("Auto-BLEU2-geometric", get_auto_bleu2_geometric),
        # ("Self-BLEU3-arithmetic", get_self_bleu3_arithmetic),
        # ("Self-BLEU3-geometric", get_self_bleu3_geometric),
        # ("Auto-BLEU3-arithmetic", get_auto_bleu3_arithmetic),
        # ("Auto-BLEU3-geometric", get_auto_bleu3_geometric),
    ]

    n_processes = min(16, len(tasks))
    with Pool(n_processes) as pool:
        id_and_metrics = pool.map(run_f, [(t[1], terms) for t in tasks])

    with open(args.output_file, "w") as f:
        for (metric_name, _), id_and_metric in zip(tasks, id_and_metrics):
            metrics = [x[1] for x in id_and_metric]
            metric, sem = np.mean(metrics), np.std(metrics) / np.sqrt(len(metrics))
            metric, sem = [round(100 * x, 2) for x in [metric, sem]]
            f.write(f"{metric_name} {metric}+-{sem}\n")

        vert = round(
            np.sqrt(
                100
                * np.mean([x[1] for x in id_and_metrics[0]])
                * 100
                * np.mean([x[1] for x in id_and_metrics[1]])
            ),
            2,
        )
        f.write(f"VERT {vert}")


def run_f(task_params):
    f, terms = task_params
    return f(terms)


if __name__ == "__main__":
    # NLTK produces warnings
    warnings.filterwarnings("ignore")

    main()
