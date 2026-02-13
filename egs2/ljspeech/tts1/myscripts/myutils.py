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


def parse_bitrate(file_path: str) -> dict[str, float]:
    """
    Input (tab-separated):
                N=20    N=40    ... N=280
        K=128   194.3   152.3   ...
        ...
    Returns:
        {"{N}-{K}": bitrate_value, ...}
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    Ns = [N_val.split("=")[1] for N_val in lines[0].strip().split("\t")]
    bitrate_dict: dict[str, float] = {}
    for line in lines[1:]:
        parts = line.strip().split("\t")
        K = parts[0].split("=")[1]
        values = list(map(float, parts[1:]))
        for N, value in zip(Ns, values):
            key = f"{N}-{K}"
            bitrate_dict[key] = value
    return bitrate_dict


def extract_llm_scores(summary_path: Path) -> list[float]:
    scores = []
    with summary_path.open("r") as f:
        next(f)  # skip header
        for line in f:
            _, score_str = line.strip().split(",")
            scores.append(float(score_str))
    return scores
