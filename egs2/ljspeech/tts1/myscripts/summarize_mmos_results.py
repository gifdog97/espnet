# *.csv ファイルを開く
from collections import defaultdict
from pathlib import Path

import numpy as np

score_dict = defaultdict(list)
for csv_file in Path("mmos_results").glob("*.csv"):
    with open(csv_file) as f:
        lines = f.readlines()
    rater_id = csv_file.stem.split("-")[1]
    for line in lines[3:]:
        _, audio_path, score = line.strip().split(",")
        audio_id = Path(audio_path).stem
        setting = audio_id.split("_")[1]  # t_20-128-0.6_LJ049-0008-0 -> 20-128-0.6
        sample_id = audio_id.split("_")[2]  # t_20-128-0.6_LJ049-0008-0 -> LJ049-0008-0
        score_dict[setting].append(f"{rater_id},{sample_id},{int(score)}")

score_dict = {setting: np.array(scores) for setting, scores in score_dict.items()}
# setting で sort
score_dict = dict(sorted(score_dict.items(), key=lambda x: x[0]))

for setting in [
    "20-128-0.6",
    "20-256-0.6",
    "20-512-0.7",
    "20-1024-0.6",
    "20-2048-0.6",
    "20-4096-0.6",
    "40-256-0.6",
    "40-512-0.7",
    "40-1024-0.6",
    "40-2048-0.6",
    "40-4096-0.7",
    "80-256-0.7",
    "80-512-0.7",
    "80-1024-0.7",
    "80-2048-0.7",
    "80-4096-0.6",
    "80-8192-0.7",
    "120-1024-0.7",
    "120-2048-0.7",
    "120-4096-0.7",
    "120-8192-0.6",
]:
    with open(f"mmos_results/summary/{setting}.csv", "w") as f:
        scores = score_dict[setting]
        f.write("rater_id,sample_id,score\n")
        for score in scores:
            f.write(f"{score}\n")
