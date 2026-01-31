"""
`dump/continuation/fixed_${N}-${K}_dedup/raw/dev/text` に以下の形式で continuation のテキストファイルを作成。
LJ049-0008-0 丑 丳 不 乹 一 丢 临 ...
"""

from pathlib import Path

import pandas as pd
from myutils import to_kanji

CONTINUATION_CSV_DIR = (
    "../../../../../speechLM/experiment/units/LJSpeech-1.1/continuation"
)
CONTINUATION_OUTPUT_DIR = "../dump/continuation"

### --- Create continuation text files for all experimental settings --- ###
# Generate settings
settings = []
for N in [20, 40, 80, 120, 160, 200, 240, 280]:
    for K in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        for temperature in [0.1 * (i + 3) for i in range(10)]:  # 0.3 to 1.2
            settings.append(f"fixed_{N}-{K}_dedup_{temperature:.1f}")

# Write continuation text files for each setting
for setting in settings:
    # Prepare inputs
    units_file = Path(f"{CONTINUATION_CSV_DIR}/{setting}.csv")
    output_path = Path(f"{CONTINUATION_OUTPUT_DIR}/{setting}/raw/dev/text")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(units_file)
    ids = df["id"].tolist()
    continuations = df["continuation"].tolist()

    # Save continuations to the output file
    eos_ng_list = []
    continuation_num = 0
    with output_path.open("w") as f:
        for wav_id, continuation in zip(ids, continuations):
            eos_split = continuation.split(" <eos>")
            if len(eos_split) == 1:
                eos_ng_list.append(wav_id)
            else:
                for i, continuation in enumerate(eos_split[:-1]):
                    # if <eos> is at the end of the continuation (i.e., split() results in ""), skip it.
                    if continuation.strip() == "":
                        continue
                    continuation_num += 1
                    f.write(f"{wav_id}-{i} {to_kanji(continuation.strip())}\n")

    # Required for TTS execution
    feats_type_file = Path(f"{CONTINUATION_OUTPUT_DIR}/{setting}/raw/dev/feats_type")
    with feats_type_file.open("w") as f:
        f.write("raw\n")

    # Save statistics
    stat_file = Path(f"{CONTINUATION_OUTPUT_DIR}/{setting}/raw/dev/stat.log")
    with stat_file.open("w") as f:
        f.write(f"Original dataset size: {len(continuations)}\n")
        f.write(f"Ids without <eos>: {eos_ng_list}\n")
        f.write(f"Final continuation dataset size: {continuation_num}\n")
