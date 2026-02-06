import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from discrete_speech_metrics import MCD, UTMOS, LogF0RMSE, SpeechBERTScore
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wav_dir",
        help="Directory of the WAV files to evaluate",
        default="../exp/fixed_20-128_dedup/tts_train_raw_phn_none/decode_with_ljspeech_style_melgan.v1/eval1/wav",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to output scores.",
        default="csv/resynthesis_dsmetrics/tacotron2-20-128.csv",
    )
    return parser.parse_args()


def create_dict_from_wav_dir(wav_dir):
    wav_dict = {}
    for filepath in Path(wav_dir).glob("*.wav"):
        key = filepath.stem  # Remove .wav extension
        wav_dict[key] = str(filepath)
    return wav_dict


def calculate_utmos_original(original_wav_dict) -> float:
    utmos_original = []
    metrics = None
    for key in original_wav_dict.keys():
        original_wav_path = original_wav_dict[key]
        original_wav, original_sr = sf.read(original_wav_path)
        if metrics is None:
            metrics = UTMOS(sr=original_sr)
        utmos_original.append(metrics.score(original_wav))
    return np.array(utmos_original).mean()


def calculate_dsmetrics(
    original_wav_dict, resynth_wav_dict
) -> tuple[float, float, float, float]:
    # Initialize metrics objects
    _, original_sr = sf.read(list(original_wav_dict.values())[0])
    speech_bert_score_metrics = SpeechBERTScore(
        sr=original_sr, model_type="wavlm-large", layer=14, use_gpu=True
    )
    mcd_metrics = MCD(sr=original_sr)
    rmse_f0_metrics = LogF0RMSE(sr=original_sr)
    utmos_metrics = UTMOS(sr=original_sr)

    # Calculate metrics
    speech_bert_score, mcd, rmse_f0, utmos = [], [], [], []
    for key in tqdm(list(original_wav_dict.keys())):
        original_wav_path = original_wav_dict[key]
        resynth_wav_path = resynth_wav_dict[key]
        original_wav, original_sr = sf.read(original_wav_path)
        resynth_wav, resynth_sr = sf.read(resynth_wav_path)
        assert original_sr == resynth_sr, "Sampling rates do not match."

        # SpeechBERTScore
        score, _, _ = speech_bert_score_metrics.score(original_wav, resynth_wav)
        speech_bert_score.append(score)

        # Cepstral Distortion (MCD)
        mcd.append(mcd_metrics.score(original_wav, resynth_wav))

        # F0 RMSE
        rmse_f0.append(rmse_f0_metrics.score(original_wav, resynth_wav))

        # UTMOS
        utmos.append(utmos_metrics.score(resynth_wav))

    return (
        np.array(speech_bert_score).mean(),
        np.array(mcd).mean(),
        np.array(rmse_f0).mean(),
        np.array(utmos).mean(),
    )


def main():
    args = get_args()
    original_wav_dir = "../data_orig/eval1/wavs"
    original_wav_dict = create_dict_from_wav_dir(original_wav_dir)
    with open("csv/resynthesis_dsmetrics/gold_utmos.csv", "w") as f:
        utmos_original = calculate_utmos_original(original_wav_dict)
        f.write("speechBERTScore,MCD,LogF0RMSE,UTMOS\n")
        f.write(f",,,{utmos_original:.4f}\n")
    with open(args.output_path, "w") as f:
        f.write("speechBERTScore,MCD,LogF0RMSE,UTMOS\n")
        resynth_wav_dir = args.wav_dir
        resynth_wav_dict = create_dict_from_wav_dir(resynth_wav_dir)
        assert set(original_wav_dict.keys()) == set(resynth_wav_dict.keys()), (
            "Mismatch in keys between original and resynthesized wav files."
        )
        speech_bert_score, mcd, rmse_f0, utmos = calculate_dsmetrics(
            original_wav_dict, resynth_wav_dict
        )
        f.write(f"{speech_bert_score:.4f},{mcd:.4f},{rmse_f0:.4f},{utmos:.4f}\n")


if __name__ == "__main__":
    main()
