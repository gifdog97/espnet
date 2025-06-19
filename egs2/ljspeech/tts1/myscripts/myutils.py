from pathlib import Path


def to_kanji(text: str) -> str:
    return " ".join([chr(0x4E00 + int(c)) for c in text.split(" ")])


def create_units_dict(units_file: Path) -> dict:
    units_dict = {}
    with units_file.open() as f:
        for line in f:
            if line.startswith("id,text"):
                continue
            wav_id, text = line.strip().split(",")
            units_dict[wav_id] = to_kanji(text)
    return units_dict
