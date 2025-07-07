from pathlib import Path

units_file = Path(
    "../../../../../speechLM/experiment/units/LJSpeech-1.1/continuation/fixed_20-128_dedup_0.3.csv"
)
gold_text_file = Path("../data_orig/dev/text")
output_file = Path("transcription/gold.txt")

ids = set()
with units_file.open("r") as f:
    for line in f:
        if line.strip():
            wav_id, _ = line.strip().split(",", 1)
            ids.add(wav_id)

with gold_text_file.open("r") as f_in, output_file.open("w") as f_out:
    for line in f_in:
        if line.strip():
            wav_id, transcription = line.strip().split(" ", 1)
            if wav_id in ids:
                f_out.write(f"{wav_id}|{transcription}\n")
