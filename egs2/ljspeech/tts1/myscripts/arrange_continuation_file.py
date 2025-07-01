"""
`dump/continuation/fixed_${N}-${K}_dedup/raw/dev/text` に以下の形式で continuation のテキストファイルを作成します。
LJ049-0008 丑 丳 不 乹 一 丢 临 ...
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
with output_file.open("w") as f:
    for wav_id, continuation in zip(ids, continuations):
        # TODO: deal with <eos> at the end of continuation
        f.write(f"{wav_id} {to_kanji(continuation[:-6].strip())}\n")

# Required for TTS execution
feats_type_file = Path(f"../dump/continuation/{args.setting}/raw/dev/feats_type")
with feats_type_file.open("w") as f:
    f.write("raw\n")
