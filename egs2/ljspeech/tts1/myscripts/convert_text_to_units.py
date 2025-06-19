import argparse
import shutil
from pathlib import Path

from myutils import create_units_dict

parser = argparse.ArgumentParser()

parser.add_argument(
    "--setting",
    help="fixed_20-128_dedup, ...",
)

args = parser.parse_args()

data_orig_dir = Path("../data_orig")
units_file = Path(
    f"/work/gk77/k77035/speechLM/experiment/units/LJSpeech-1.1/wavs/{args.setting}.csv"
)


units_dict = create_units_dict(units_file)

output_root_dir = Path("../data")

# data_root_dir内の`text`という名前のファイルを再帰的に取得
for text_file in data_orig_dir.glob("**/text"):
    output_dir = output_root_dir.joinpath(units_file.stem, text_file.parent.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    # output_dir に text_file.parent の中のファイルをcopy
    for file in text_file.parent.iterdir():
        shutil.copy(file, output_dir)
    with open(text_file) as srcf, open(output_dir / "text", "w") as tgtf:
        for line in srcf:
            wav_id = line.strip().split(" ")[0]
            tgtf.write(f"{wav_id} {units_dict[wav_id]}\n")
