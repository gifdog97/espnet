import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

parser = argparse.ArgumentParser()

parser.add_argument(
    "--wav_dir",
    help="Directory of the WAV files to transcribe",
    default="/work/gk77/k77035/espnet/egs2/ljspeech/tts1/exp/fixed_20-128_dedup/tts_train_raw_phn_none/continuation_0.3/dev/wav",
)
parser.add_argument(
    "--output_path",
    help="Output path for the transcriptions",
    default="/work/gk77/k77035/espnet/egs2/ljspeech/tts1/myscripts/transcription/fixed_20-128-0.3.txt",
)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

with open(args.output_path, "w") as f:
    for wav_file in sorted(Path(args.wav_dir).glob("*.wav")):
        result = pipe(str(wav_file), generate_kwargs={"language": "english"})
        text: str = result["text"]
        f.write(f"{wav_file.stem}|{text.strip()}\n")
