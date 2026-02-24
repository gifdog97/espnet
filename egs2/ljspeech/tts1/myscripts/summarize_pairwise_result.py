import argparse
import os
import re
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

PAIRWISE_DIR = Path("pairwise_10s")
SCORE_MAP = {"A>>B": 1, "A>B": 0.5, "A=B": 0, "B>A": -0.5, "B>>A": -1}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairwise_dir",
        type=Path,
        default=PAIRWISE_DIR,
        help="Directory containing pairwise LLM evaluation results",
    )
    return parser.parse_args()


def parse_llm_output(
    llm_output_path: Path, setting_X: str
) -> tuple[float, int, int, int]:
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
    text_A_length = len(content[1].strip().split(" "))
    text_B_length = len(content[3].strip().split(" "))
    length_diff = text_A_length - text_B_length
    return score, text_A_length, text_B_length, length_diff


def write_summary(pairwise_dir: Path):
    setting_X, _ = pairwise_dir.name.split("_vs_")
    summary_dict = {
        "utt_id": [],
        "score": [],
        "length_A": [],
        "length_B": [],
        "length_diff": [],
    }
    for llm_output in pairwise_dir.iterdir():
        if "summary" in llm_output.name:
            continue
        if not llm_output.name.endswith(".txt"):
            continue
        utt_id = llm_output.stem
        try:
            score, text_A_length, text_B_length, length_diff = parse_llm_output(
                llm_output, setting_X
            )
        except ValueError as e:
            print(e)
            continue
        summary_dict["utt_id"].append(utt_id)
        summary_dict["score"].append(score)
        summary_dict["length_A"].append(text_A_length)
        summary_dict["length_B"].append(text_B_length)
        summary_dict["length_diff"].append(length_diff)
    summary_path = pairwise_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write("utt_id,score,length_A,length_B,length_diff\n")
        for i in range(len(summary_dict["utt_id"])):
            f.write(
                f"{summary_dict['utt_id'][i]},{summary_dict['score'][i]},{summary_dict['length_A'][i]},{summary_dict['length_B'][i]},{summary_dict['length_diff'][i]}\n"
            )
    return summary_dict


def main():
    args = parse_args()
    cpu_count = os.cpu_count()
    pairwise_dirs = []
    for pairwise_dir in Path(args.pairwise_dir).iterdir():
        if "_vs_" not in pairwise_dir.name:
            continue
        pairwise_dirs.append(pairwise_dir)
    with Pool(cpu_count) as pool, tqdm(total=len(pairwise_dirs)) as pbar:
        for _ in pool.imap(write_summary, pairwise_dirs):
            pbar.update(1)


if __name__ == "__main__":
    main()
