import argparse
import math

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
                # Use first sentence only
                # transcript_dict[current_id] += _postprocess(transcript)
                continue
            current_id = "-".join(wav_id.split("-")[:2])
            transcript_dict[current_id] = _postprocess(transcript)
    return {k: v.strip() for k, v in transcript_dict.items()}


if __name__ == "__main__":
    args = get_args()
    # gold_dict = load_transcript("transcription/gold.txt")
    continuation_dict = load_transcript(args.asr_transcript)
    model_id = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    model.cuda()
    ppls = {}
    total_nll = 0
    total_length = 0
    for wav_id, text in continuation_dict.items():
        # if wav_id not in gold_dict:
        #     print(f"Skipping {wav_id} as it is not in the gold transcript.")
        #     ppls[wav_id] = None
        #     continue
        # gold_inputs = tokenizer(gold_dict[wav_id], return_tensors="pt")
        inputs = tokenizer(text, return_tensors="pt")
        # gold_length = gold_inputs["input_ids"].size(1)
        # if inputs["input_ids"].size(1) < gold_length:
        #     ppls[wav_id] = None
        #     print(f"Skipping {wav_id} due to short input length.")
        #     continue
        # inputs = {k: v[:, :gold_length].cuda() for k, v in inputs.items()}  # GPUへ送る
        inputs = {k: v.cuda() for k, v in inputs.items()}  # GPUへ送る
        # ラベルを自分自身にする（自己回帰モデル）
        labels = inputs["input_ids"]
        # 損失計算（=負の対数尤度の平均）
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            nll = outputs.loss
        total_nll += nll.item() * inputs["input_ids"].size(1)
        total_length += inputs["input_ids"].size(1)
        # 文ごとのPPLを計算
        ppl = torch.exp(nll)
        ppls[wav_id] = ppl.item()
    with open(args.output_file, "w") as f:
        for wav_id, ppl in ppls.items():
            f.write(f"{wav_id}|{ppl:.3f}\n")
        ppl_corpus = math.exp(total_nll / total_length)
        f.write(f"PPL_corpus|{ppl_corpus:.3f}\n")
