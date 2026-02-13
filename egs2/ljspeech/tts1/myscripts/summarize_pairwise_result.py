import re
from pathlib import Path

PAIRWISE_DIR = Path("pairwise")
SCORE_MAP = {"A>>B": 1, "A>B": 0.5, "A=B": 0, "B>A": -0.5, "B>>A": -1}


def parse_llm_output(llm_output_path: Path, setting_X: str) -> str:
    with llm_output_path.open("r") as f:
        content = [line.rstrip("\n") for line in f if line.strip()]
    # Text A (vits-20-128-0.7) -> vits-20-128-0.7
    text_A_setting = content[0].strip().split(" ")[-1].strip("()")
    # [[judge]] -> judge
    judge_found = re.findall(r"\[\[(.*?)\]\]", content[-1])
    if not judge_found:
        raise ValueError(f"Could not find judge in {llm_output_path}")
    judge = judge_found[0]
    if text_A_setting == setting_X:  # A and B is not swapped
        score = SCORE_MAP[judge]
    else:  # A and B is swapped
        score = -SCORE_MAP[judge]
    return score


def write_summary(pairwise_dir: Path):
    setting_X, _ = pairwise_dir.name.split("_vs_")
    summary_lines = []
    for llm_output in pairwise_dir.iterdir():
        if not llm_output.name.endswith(".txt"):
            continue
        utt_id = llm_output.stem
        try:
            judge = parse_llm_output(llm_output, setting_X)
        except ValueError as e:
            print(e)
            continue
        summary_lines.append(f"{utt_id},{judge}")
    summary_path = pairwise_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write("utt_id,score\n")
        for line in summary_lines:
            f.write(line + "\n")


def main():
    for pairwise_dir in PAIRWISE_DIR.iterdir():
        if "_vs_" not in pairwise_dir.name:
            continue
        if (pairwise_dir / "summary.txt").exists():
            print(f"summary.txt already exists in {pairwise_dir}, skipping...")
            continue
        print(pairwise_dir)
        write_summary(pairwise_dir)


if __name__ == "__main__":
    main()
