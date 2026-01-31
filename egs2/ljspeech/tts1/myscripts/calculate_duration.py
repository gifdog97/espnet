import csv
import wave
from pathlib import Path

import pandas as pd


def extract_settings(result_tsv: str):
    settings = []
    with open(result_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["temperature"] == "None":
                continue
            setting, temperature = row["setting"], float(row["temperature"])
            N, K = setting.split("-")
            settings.append((N, K, temperature))
    return settings


def print_duration(result_tsv: str, path_template: str, output_csv: str):
    settings = extract_settings(result_tsv)
    wav_dirs = [
        Path(path_template.format(N=N, K=K, temperature=temperature))
        for N, K, temperature in settings
    ]
    result_dict = {
        "setting": [],
        "total_utterances": [],
        "total_duration_sec": [],
        "average_duration_sec": [],
    }
    for setting, wav_dir in zip(settings, wav_dirs):
        N, K, temperature = setting
        total_duration = 0.0
        total_utterances = 0
        for wav_path in wav_dir.glob("*.wav"):
            if not wav_path.stem.endswith("-0"):
                continue
            with wave.open(str(wav_path), mode="rb") as wf:
                duration = float(wf.getnframes() / wf.getframerate())
            total_duration += duration
            total_utterances += 1
        result_dict["setting"].append(f"{N}-{K}-{temperature}")
        result_dict["total_utterances"].append(total_utterances)
        result_dict["total_duration_sec"].append(f"{total_duration:.2f}")
        result_dict["average_duration_sec"].append(
            f"{total_duration / total_utterances:.2f}"
        )
    df = pd.DataFrame(result_dict)
    df.to_csv(output_csv, index=False)


print("===Tacotron2===")
print_duration(
    "csv/continuation_result.tsv",
    "../exp/fixed_{N}-{K}_dedup/tts_train_raw_phn_none/continuation_{temperature}/dev/wav",
    "csv/duration_results.csv",
)

print("===VITS===")
print_duration(
    "csv/continuation_result-vits.tsv",
    "../exp/fixed_{N}-{K}_dedup-vits/tts_train_vits_raw_phn_none/vits_continuation_{temperature}/dev/wav",
    "csv/duration_results-vits.csv",
)
