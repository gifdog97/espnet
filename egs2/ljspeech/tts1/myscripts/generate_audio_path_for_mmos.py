import csv
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


def assign_samples_to_worker():
    settings_t = extract_settings("csv/continuation_result.tsv")
    path_template_t = "/work/gk77/k77035/espnet/egs2/ljspeech/tts1/exp/fixed_{N}-{K}_dedup/tts_train_raw_phn_none/continuation_{temperature}/dev/wav"
    wav_dirs_t = [
        Path(path_template_t.format(N=N, K=K, temperature=temperature))
        for N, K, temperature in settings_t
    ]
    settings_v = extract_settings("csv/continuation_result-vits.tsv")
    path_template_v = "/work/gk77/k77035/espnet/egs2/ljspeech/tts1/exp/fixed_{N}-{K}_dedup-vits/tts_train_vits_raw_phn_none/vits_continuation_{temperature}/dev/wav"
    wav_dirs_v = [
        Path(path_template_v.format(N=N, K=K, temperature=temperature))
        for N, K, temperature in settings_v
    ]
    wav_dirs = wav_dirs_t + wav_dirs_v
    worker_to_wav_paths = {str(i).zfill(3): [] for i in range(1, 101)}
    # length of wavdirs is equal to the number of settings
    for i, wav_dir in enumerate(wav_dirs):
        stimuli_paths = []
        for wav_path in sorted(wav_dir.glob("*.wav")):
            if not wav_path.stem.endswith("-0"):
                continue
            stimuli_paths.append(wav_path)
            # sample 100 audio files
            if len(stimuli_paths) == 100:
                break
        # 100 stimuli rotated by i
        stimuli_rotated = stimuli_paths[i:] + stimuli_paths[:i]
        for j in range(100):
            worker_id = str(j + 1).zfill(3)
            worker_to_wav_paths[worker_id].append(str(stimuli_rotated[j].resolve()))
    df = pd.DataFrame.from_dict(worker_to_wav_paths, orient="index").transpose()
    df.to_csv("csv/worker_to_wav_paths.csv", index=False)


if __name__ == "__main__":
    assign_samples_to_worker()
