import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asr_transcript",
        type=str,
        help="Path to the transcript file.",
        default="transcription/fixed_20-128-0.3.txt",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to output scores.",
        default="csv/ppl/fixed_20-128-0.3.txt",
    )
    return parser.parse_args()


def load_transcript(transcript_path: str) -> dict[str, str]:
    def _postprocess(transcript: str) -> str:
        # capitalize the first letter, add punctuation to the end if needed
        if not transcript.endswith("."):
            transcript += "."
        return transcript[0].upper() + transcript[1:] + " "

    transcript_dict = {}
    current_id = "dummy"
    with open(transcript_path) as f:
        for line in f:
            wav_id, transcript = line.strip().split("|")
            if wav_id.startswith(current_id):
                transcript_dict[current_id] += _postprocess(transcript)
                continue
            current_id = "-".join(wav_id.split("-")[:2])
            transcript_dict[current_id] = _postprocess(transcript)
    return {k: v.strip() for k, v in transcript_dict.items()}


if __name__ == "__main__":
    args = get_args()
    gold_dict = load_transcript("transcription/gold.txt")
    continuation_dict = load_transcript(args.asr_transcript)
    model_id = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    model.cuda()
    ppls = {}
    for wav_id, text in continuation_dict.items():
        gold_inputs = tokenizer(gold_dict[wav_id], return_tensors="pt")
        inputs = tokenizer(text, return_tensors="pt")
        gold_length = gold_inputs["input_ids"].size(1)
        if inputs["input_ids"].size(1) < gold_length:
            ppls[wav_id] = None
            print(f"Skipping {wav_id} due to short input length.")
            continue
        inputs = {k: v[:, :gold_length].cuda() for k, v in inputs.items()}  # GPUへ送る
        # ラベルを自分自身にする（自己回帰モデル）
        labels = inputs["input_ids"]
        # 損失計算（=負の対数尤度の平均）
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            neg_log_likelihood = outputs.loss
        # Perplexity = exp(平均負対数尤度)
        ppl = torch.exp(neg_log_likelihood)
        ppls[wav_id] = ppl.item()
    with open(args.output_file, "w") as f:
        for wav_id, ppl in ppls.items():
            if ppl is not None:
                f.write(f"{wav_id}|{ppl:.3f}\n")
            else:
                f.write(f"{wav_id}|None\n")
        ppl_without_none = {k: v for k, v in ppls.items() if v is not None}
        ppl_values = np.array(list(ppl_without_none.values()))
        f.write(f"Average|{np.mean(ppl_values):.3f}\n")
        f.write(f"Median|{np.median(ppl_values):.3f}\n")
