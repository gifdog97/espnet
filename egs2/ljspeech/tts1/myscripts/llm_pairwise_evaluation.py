import argparse
import csv
import itertools
import random
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from string import Template
from typing import Union

from openai import OpenAI, RateLimitError
from tqdm import tqdm

with open("secret.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("OPENAI_API_KEY="):
            api_key = line.strip().split("=")[1]
        if line.startswith("OPENAI_API_BASE="):
            api_base = line.strip().split("=")[1]
client = OpenAI(
    api_key=api_key,
    base_url=api_base,
)

PROMPT_TEMPLATE = Template("""
# Instructions

Please act as an impartial judge and evaluate the quality of two texts which occur in the context of a book. These texts are transcribed from audio recordings that were truncated to a fixed duration. Your job is to consider the following criteria to evaluate which text is better:

- Fluency: How grammatically correct is the text?
- Coherence: How well do the sentences of the text fit together?
- Logicality: How much does the text obey common sense?

First, read text A and consider its fluency, coherence, and logicality. Do not penalize the text for ending mid-sentence or mid-paragraph.

Then, read text B and consider its fluency, coherence, and logicality. Do not penalize the text for ending mid-sentence or mid-paragraph.

Afterwards, compare the fluency and coherence of the two texts. Do not penalize either text for ending mid-sentence or mid-paragraph.

Finally, after providing your explanations, you must output only one of the following choices as your final verdict with a label:
1. Text A is significantly better: [[A>>B]]
2. Text A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Text B is slightly better: [[B>A]]
5. Text B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]".

# Comparison task
 
## ---------- Text A ----------
${text_A}

## ---------- Text B ----------
${text_B}

## ---------- Detailed Comparison of Continuations ----------
""")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pairwise evaluation of LLM-generated continuations."
    )
    parser.add_argument(
        "--continuation_results",
        type=str,
        default="csv/continuation_result_10s.csv",
        help="Path to the CSV file containing continuation results.",
    )
    parser.add_argument(
        "--transcription_dir",
        type=str,
        default="transcription_cut_10s",
        help="Path to the directory containing transcription files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pairwise_10s",
        help="Directory to save pairwise comparison results.",
    )
    return parser.parse_args()


def extract_transcriptions(file_path: Path):
    transcriptions = {}
    with open(file_path) as f:
        for line in f:
            wav_id, transcription = line.strip().split("|")
            # NOTE: Instead of extracting first utterance with "-0" suffix, we turn to extract 10s trimmed transcriptions.
            # if not wav_id.endswith("-0"):
            #     continue
            transcriptions[wav_id] = transcription.strip()
    return transcriptions


def with_retry(func, *args, max_retries=5, **kwargs):
    for i in range(max_retries):
        try:
            return func(*args, **kwargs)
        except RateLimitError:
            wait = 2**i + random.random()
            print(f"RateLimit hit, retrying in {wait:.1f} sec...")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")


def calculate_score(
    setting_transcriptions_X: dict, setting_transcriptions_Y: dict, output_dir: Path
) -> dict[str, Union[int, float]]:
    """
    Quality score of X over Y judged by GPT-4.1-mini.
    Score is between -1 and 1, and if X is better than Y, score is positive.
    """
    setting_X = setting_transcriptions_X["setting"]
    setting_Y = setting_transcriptions_Y["setting"]
    print(f"{setting_X} vs {setting_Y}")
    output_dir = output_dir / f"{setting_X}_vs_{setting_Y}"
    output_dir.mkdir(parents=True, exist_ok=True)
    transcriptions_X = setting_transcriptions_X["transcription"]
    transcriptions_Y = setting_transcriptions_Y["transcription"]
    score_map = {"A>>B": 1, "A>B": 0.5, "A=B": 0, "B>A": -0.5, "B>>A": -1}
    total_count = 0
    total_score = 0
    for wav_id, transcription_x in tqdm(transcriptions_X.items()):
        transcription_y = transcriptions_Y.get(wav_id)
        if not transcription_y:
            continue
        total_count += 1
        # flip X and Y with 50% probability
        ab = random.random() < 0.5
        if ab:
            prompt = PROMPT_TEMPLATE.substitute(
                text_A=transcription_x, text_B=transcription_y
            )
        else:
            prompt = PROMPT_TEMPLATE.substitute(
                text_A=transcription_y, text_B=transcription_x
            )
        completion = with_retry(
            client.chat.completions.create,
            model="GPT-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        completion_text = completion.choices[0].message.content
        assert isinstance(completion_text, str)
        with (output_dir / f"{wav_id}.txt").open("w") as f:
            if ab:
                f.write(f"Text A ({setting_X})\n")
                f.write(transcription_x + "\n")
                f.write(f"Text B ({setting_Y})\n")
                f.write(transcription_y + "\n")
            else:
                f.write(f"Text A ({setting_Y})\n")
                f.write(transcription_y + "\n")
                f.write(f"Text B ({setting_X})\n")
                f.write(transcription_x + "\n")
            f.write("\n")
            f.write(completion_text)
        judge = re.findall(r"\[\[(.*?)\]\]", completion_text.strip().split("\n")[-1])[0]
        if ab:
            score = score_map[judge]
        else:
            score = -score_map[judge]
        total_score += score
    return {
        "setting_X": setting_X,
        "setting_Y": setting_Y,
        "total_count": int(total_count),
        "total_score": float(total_score),
        "average_score": float(total_score / total_count),
    }


def main():
    args = parse_args()
    with open(Path(args.continuation_results), "r") as f:
        reader = csv.DictReader(f)
        valid_settings = [
            f"{row['setting']}-{row['temperature']}"
            for row in reader
            if row["temperature"] != ""
        ]
    results = []
    max_workers = 8
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        future_to_pair = {}
        # アドホックに一部をリトライしたい場合:
        # for setting_X, setting_Y in [
        #     ("tacotron2-40-256-0.6", "vits-120-8192-0.6"),
        # ]:
        # 全設定をやる場合:
        for setting_X, setting_Y in itertools.product(valid_settings, repeat=2):
            model_X, nkt_X = setting_X.split("-", 1)
            model_Y, nkt_Y = setting_Y.split("-", 1)
            # 異モデルの比較はスキップ
            if model_X != model_Y:
                continue
            # 存在するペアの再計算を避ける
            if (
                Path(args.output_dir) / f"{setting_X}_vs_{setting_Y}" / "summary.txt"
            ).exists():
                print(
                    f"Output for pair ({setting_X}, {setting_Y}) already exists. Skipping..."
                )
                continue
            transcription_X = extract_transcriptions(
                Path(f"{args.transcription_dir}/{model_X}/{nkt_X}.txt")
            )
            transcription_Y = extract_transcriptions(
                Path(f"{args.transcription_dir}/{model_Y}/{nkt_Y}.txt")
            )
            dict_X = {"setting": setting_X, "transcription": transcription_X}
            dict_Y = {"setting": setting_Y, "transcription": transcription_Y}
            future = executor.submit(
                calculate_score, dict_X, dict_Y, Path(args.output_dir)
            )
            future_to_pair[future] = (setting_X, setting_Y)
            futures.append(future)
        for future in as_completed(futures):
            setting_X, setting_Y = future_to_pair[future]
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error for pair ({setting_X}, {setting_Y}): {e}")
                traceback.print_exc()
    # Set UID to avoid overwriting results, particularly when retrying a subset of pairs.
    uid = time.strftime("%Y%m%d-%H%M%S")
    with open(Path(args.output_dir) / f"result_summary_{uid}.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        for result in results:
            writer.writerow(result)


if __name__ == "__main__":
    main()
