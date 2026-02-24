import itertools
import os
import wave
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from myutils import is_valid_setting
from pydub import AudioSegment
from tqdm import tqdm


# Ns = ["20", "40", "80", "120", "160", "200", "240", "280"]
def extract_valid_settings(resynthesis_result_file: str):
    df = pd.read_csv(resynthesis_result_file, index_col=0)
    NK_combinations = list(
        itertools.product(
            [20, 40, 80, 120, 160, 200, 240, 280], [2**i for i in range(7, 7 + 8)]
        )
    )
    valid_settings = []
    for model in ["tacotron2", "vits"]:
        for N, K in NK_combinations:
            setting = f"{model}-{N}-{K}"
            wer = float(df.loc[setting, "wer"])
            utmos = float(df.loc[setting, "UTMOS"])
            if is_valid_setting(wer, utmos):
                valid_settings.append(setting)
    return valid_settings


temperatures = ["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.1", "1.2"]


def build_tasks(valid_settings):
    tasks = []
    for setting in valid_settings:
        model, NK = setting.split("-", 1)
        for temperature in temperatures:
            if model == "tacotron2":
                wav_path = (
                    f"/work/gk77/k77035/espnet/egs2/ljspeech/tts1/exp/"
                    f"fixed_{NK}_dedup/tts_train_raw_phn_none/"
                    f"continuation_{temperature}/dev/wav"
                )
            else:  # vits
                wav_path = (
                    f"/work/gk77/k77035/espnet/egs2/ljspeech/tts1/exp/"
                    f"fixed_{NK}_dedup-vits/tts_train_vits_raw_phn_none/"
                    f"vits_continuation_{temperature}/dev/wav"
                )
            wav_files = sorted(Path(wav_path).glob("*.wav"))
            tasks.extend((f"{setting}-{temperature}", wf) for wf in wav_files)
    return tasks


def duration_task(args):
    """Process worker: compute duration for one wav file."""
    setting, wav_file = args
    wav_id = wav_file.stem
    with wave.open(str(wav_file), "r") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
    duration = frames / float(rate)
    return setting, wav_id, duration, str(wav_file)


def extract_lessthan10_settings(result_df):
    df = result_df.copy()
    df["utt_id"] = df["wav_id"].apply(lambda x: "-".join(x.split("-")[:-1]))
    df = df.groupby(["setting", "utt_id"], as_index=False)["duration"].sum()
    lessthan10_settings = list(df[df["duration"] < 10]["setting"])
    counter = Counter(lessthan10_settings)
    """
    Tacotron2:
    {'20-128-0.3': 10, '120-1024-0.3': 6, '20-256-0.3': 6, '40-1024-0.3': 6, '80-256-0.3': 6, '20-2048-0.3': 5, '20-128-0.4': 4,
    '40-256-0.3': 4, '20-2048-0.4': 3, '20-256-0.4': 3, '20-4096-0.3': 3, '20-4096-0.4': 3, '40-1024-0.4': 3, '40-1024-0.9': 3,
    '40-1024-1.1': 3, '40-256-0.4': 3, '40-512-0.4': 3, '120-4096-0.3': 2, '120-8192-0.3': 2, '20-1024-0.3': 2, '20-4096-1.0': 2,
    '20-4096-1.2': 2, '40-1024-0.5': 2, '40-512-0.3': 2, '80-2048-0.3': 2, '20-1024-0.5': 1, '20-1024-1.1': 1, '20-128-0.5': 1,
    '20-128-0.6': 1, '20-128-1.0': 1, '20-2048-0.5': 1, '20-2048-1.1': 1, '20-256-0.5': 1, '20-256-1.2': 1, '20-4096-0.5': 1,
    '20-4096-0.7': 1, '20-4096-1.1': 1, '20-512-0.3': 1, '20-512-0.4': 1, '20-512-0.5': 1, '40-1024-0.6': 1, '40-4096-0.5': 1,
    '40-4096-0.8': 1, '40-512-1.2': 1, '80-2048-0.5': 1, '80-256-0.4': 1, '80-512-0.3': 1, '80-8192-0.3': 1}
    VITS:
    {'vits-20-128-0.3': 10, 'vits-80-16384-0.3': 6, 'vits-20-256-0.3': 5, 'vits-20-128-0.4': 4, 'vits-20-2048-0.3': 4,
    'vits-20-8192-0.3': 4, 'vits-20-2048-0.4': 3, 'vits-20-4096-0.4': 3, 'vits-40-256-0.3': 3, 'vits-20-1024-0.3': 2,
    'vits-20-1024-0.5': 2, 'vits-20-256-0.4': 2, 'vits-20-4096-0.3': 2, 'vits-20-8192-0.4': 2, 'vits-40-1024-0.3': 2,
    'vits-40-1024-0.4': 2, 'vits-40-256-0.4': 2, 'vits-40-512-0.3': 2, 'vits-40-512-0.4': 2, 'vits-40-8192-0.4': 2,
    'vits-80-16384-0.4': 2, 'vits-80-2048-0.3': 2, 'vits-20-128-0.6': 1, 'vits-20-128-1.0': 1, 'vits-20-256-0.5': 1,
    'vits-20-256-1.2': 1, 'vits-20-4096-0.7': 1, 'vits-20-512-0.3': 1, 'vits-20-512-0.4': 1, 'vits-20-512-0.5': 1,
    'vits-20-8192-0.6': 1, 'vits-20-8192-1.1': 1, 'vits-40-1024-0.5': 1, 'vits-40-128-0.3': 1, 'vits-40-128-0.4': 1,
    'vits-40-128-0.5': 1, 'vits-40-512-1.2': 1, 'vits-40-8192-0.3': 1, 'vits-80-16384-0.5': 1, 'vits-80-512-0.3': 1,
    'vits-80-8192-0.3': 1})
    """
    return counter


def duration_stats():
    valid_settings = extract_valid_settings("csv/resynthesis_result.csv")
    tasks = build_tasks(valid_settings)
    max_workers = os.cpu_count()
    settings = []
    wav_ids = []
    durations = []
    wav_paths = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        # executor.map は入力順を保つ。tqdm で進捗表示。
        for setting, wav_id, dur, wav_file in tqdm(
            ex.map(duration_task, tasks), total=len(tasks)
        ):
            settings.append(setting)
            wav_ids.append(wav_id)
            durations.append(dur)
            wav_paths.append(wav_file)

    return pd.DataFrame(
        {
            "setting": settings,
            "wav_id": wav_ids,
            "duration": durations,
            "wav_path": wav_paths,
        }
    )


def cut_and_save(result_df):
    """
    Take sums of duration based on the prefix of wav_id (e.g., LJ049-0008)
    input_df:
    wav_id  duration
    LJ049-0008-0   4.446621
    LJ049-0008-1   9.613061
    output_df:
    wav_id  duration
    LJ049-0008   14.059682
    """
    df = result_df.copy()
    df["utt_id"] = df["wav_id"].apply(lambda x: "-".join(x.split("-")[:-1]))
    groupby_df = df.groupby(["setting", "utt_id"], as_index=False)
    for (setting, utt_id), group in groupby_df:
        output_path = Path(
            f"/work/gk77/k77035/espnet/egs2/ljspeech/tts1/myscripts/audio_cut_10s/{setting}/{utt_id}.wav"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # 空のAudioSegmentを用意
        combined = AudioSegment.empty()
        # 順番に連結
        for path in group["wav_path"].tolist():
            audio = AudioSegment.from_file(path)
            combined += audio
        if combined.duration_seconds < 10:
            print(
                f"Warning: {setting} {utt_id} is shorter than 10 seconds ({combined.duration_seconds:.2f} seconds). Skipping cut and save."
            )
            continue
        # 前半10秒（pydubはミリ秒単位）
        first_10_sec = combined[:10_000]
        # 保存
        first_10_sec.export(output_path, format="wav")
    return df


if __name__ == "__main__":
    df = duration_stats()
    # print(extract_lessthan10_settings(df))
    cut_and_save(df)
