import itertools
import re

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

OUT_PATH = "csv/resynthesis_error_rates.csv"


def compute_error_rates(
    ref: dict[str, str], hyp: dict[str, str]
) -> tuple[float, float]:
    normalizer = EnglishTextNormalizer()
    ref_list, hyp_list = [], []
    for wav_id in ref.keys():
        ref_list.append(normalizer(ref[wav_id]))
        hyp_list.append(normalizer(hyp[wav_id]))
    wer = jiwer.wer(ref_list, hyp_list)
    cer = jiwer.cer(ref_list, hyp_list)
    return wer * 100, cer * 100


def parse_transcriptions(transcription_path: str) -> dict[str, str]:
    with open(transcription_path, "r") as f:
        results = {}
        for line in f:
            wav_id, transcription = re.split("[| ]", line.strip(), maxsplit=1)
            results[wav_id] = transcription
    return results


def main():
    original_transcription_path = (
        "/work/gk77/k77035/espnet/egs2/ljspeech/tts1/data_orig/eval1/text"
    )
    original_results = parse_transcriptions(original_transcription_path)
    NK_combinations = list(
        itertools.product(
            [20, 40, 80, 120, 160, 200, 240, 280], [2**i for i in range(7, 7 + 8)]
        )
    )
    gold_transcription_path = "transcription/resynthesis/gold.txt"
    gold_results = parse_transcriptions(gold_transcription_path)
    assert set(original_results.keys()) == set(gold_results.keys())
    with open(OUT_PATH, "w") as f:
        f.write("Model,N,K,WER,CER\n")
        wer, cer = compute_error_rates(original_results, gold_results)
        f.write(f"Gold,,,{wer:.2f},{cer:.2f}\n")
        for model in ["tacotron2", "vits"]:
            for N, K in NK_combinations:
                transcription_path = (
                    f"transcription/resynthesis/{model}/fixed_{N}-{K}.txt"
                )
                resynthesis_results = parse_transcriptions(transcription_path)
                assert set(original_results.keys()) == set(resynthesis_results.keys())
                wer, cer = compute_error_rates(original_results, resynthesis_results)
                f.write(f"{model},{N},{K},{wer:.2f},{cer:.2f}\n")


if __name__ == "__main__":
    main()
