"""_summary_
For performing TTS, we need to convert original text files to unit representation and save them in `data` directory.
All stages in espnet TTS recipe refer to `data` directory (./run.sh --datadir <datadir>).
This script copies necessary files from `data_orig` to `data/{setting}/{subset}`,
and convert original text files to unit representation in kanji, using unit csv files.
"""

import argparse
import shutil
from pathlib import Path

from myutils import create_units_dict

parser = argparse.ArgumentParser()
parser.add_argument(
    "--units_dir", default="speechLM/experiment/units/LJSpeech-1.1/wavs"
)

args = parser.parse_args()

data_orig_dir = Path("../data_orig")
output_root_dir = Path("../data")

for N in [20, 40, 80, 120, 160, 200, 240, 280]:
    for i in range(7, 15):
        K = 2**i
        setting = f"fixed_{N}-{K}_dedup"
        units_file = Path(f"{args.units_dir}/{setting}.csv")
        units_dict = create_units_dict(units_file)
        # data_orig_dir 内の`text`という名前のファイルを再帰的に取得
        for text_file in data_orig_dir.glob("**/text"):
            output_dir = output_root_dir.joinpath(
                units_file.stem, text_file.parent.name
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            # output_dir に text_file.parent の中のファイルをcopy
            for file in text_file.parent.iterdir():
                shutil.copy(file, output_dir)
            with open(text_file) as srcf, open(output_dir / "text", "w") as tgtf:
                for line in srcf:
                    wav_id = line.strip().split(" ")[0]
                    tgtf.write(f"{wav_id} {units_dict[wav_id]}\n")
