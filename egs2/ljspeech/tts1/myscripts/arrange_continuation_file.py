"""
`dump/continuation/fixed_${N}-${K}_dedup/raw/dev/text` に以下の形式で continuation のテキストファイルを作成。
LJ049-0008-0 丑 丳 不 乹 一 丢 临 ...
"""

import argparse
from pathlib import Path

import pandas as pd
from myutils import to_kanji

parser = argparse.ArgumentParser()

parser.add_argument(
    "--setting",
    help="fixed_20-128_dedup, ...",
    default="fixed_20-128_dedup",
)

args = parser.parse_args()

units_file = Path(
    f"../../../../../speechLM/experiment/units/LJSpeech-1.1/continuation/{args.setting}.csv"
)

output_file = Path(f"../dump/continuation/{args.setting}/raw/dev/text")
output_file.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(units_file)
ids = df["id"].tolist()
continuations = df["continuation"].tolist()

# Save continuations to the output file
eos_ng_list = []
continuation_num = 0
with output_file.open("w") as f:
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
feats_type_file = Path(f"../dump/continuation/{args.setting}/raw/dev/feats_type")
with feats_type_file.open("w") as f:
    f.write("raw\n")

# Save statistics
stat_file = Path(f"../dump/continuation/{args.setting}/raw/dev/stat.log")
with stat_file.open("w") as f:
    f.write(f"Original dataset size: {len(continuations)}\n")
    f.write(f"Ids without <eos>: {eos_ng_list}\n")
    f.write(f"Final continuation dataset size: {continuation_num}\n")
