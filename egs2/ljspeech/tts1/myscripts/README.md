# On the Effect of Segmentation Width and Cluster Size on Speech Resynthesis and Continuation in Generative Spoken Language Models

This directory provides the codebase for experiments conducted in the present paper (published in INTERSPEECH 2026).
We forked [ESPNet](https://github.com/espnet/espnet) and implemented the experimental codebase on top of it.

To get started, make sure to setup your python environment.
The code assumes python version `3.9.7` and setup with `venv` (see [espnet installation page](https://espnet.github.io/espnet/installation.html) for reference.).

## 1. Download audio files and create `data_orig` directory (LJSpeech)

Run following command that uses the official script provided by espnet.
Note that our experiment covers multiple N,K values, where N is taken from `{20, 40, 80, 120, 160, 200, 240, 280}` and K is from `{128, 256, 512, 1024, 2048, 4096, 8192, 16384}`.

```
./run.sh --stage 1 --stop-stage 1
```

## 2. Create datadir and prepare for experimental files

The default pipeline of ESPnet TTS prepares for `text` files that include normal texts to be synthesized.
In GSLM framework, we synthesize speech from "units", discrete representations from SSL models.
Therefore, we need to replace texts in `text` into units.

To begin with, refer to [speechLM repository](https://github.com/mynlp/speechLM) and prepare for units file.
Then, run following command specifying `units_dir` where units file are stored.

```
python arrange_datadir.py --units_dir <units_dir>
```

Run following command, then it's ready for training unit2speech model with Tacotron2.

```
./run.sh --stage 2 --stop-stage 6 \
    --datadir data/fixed_${N}-${K}_dedup \
    --dumpdir dump/fixed_${N}-${K}_dedup \
    --expdir exp/fixed_${N}-${K}_dedup \
    --nj 16
```


If you train VITS, run following instead.

```
/run.sh --stage 2 --stop-stage 6 \
    --train_config conf/tuning/train_vits.yaml
    --inference_config conf/tuning/decode_vits.yaml
    --tts_task gan_tts
    --datadir data/fixed_${N}-${K}_dedup \
    --dumpdir dump/fixed_${N}-${K}_dedup \
    --expdir exp/fixed_${N}-${K}_dedup-vits \
    --nj 16
```

## 3. Train unit2speech

Train unit2speech models with Tacotron2 using following command.

```
./run.sh --stage 7 --stop-stage 7 \
    --datadir data/fixed_${N}-${K}_dedup \
    --dumpdir dump/fixed_${N}-${K}_dedup \
    --expdir exp/fixed_${N}-${K}_dedup
```

If you train VITS, additionally specify the following related options.

```
    --train_config conf/tuning/train_vits.yaml \
    --inference_config conf/tuning/decode_vits.yaml \
    --tts_task gan_tts \
```

## 4. Decode with unit2speech

### Speech resynthesis

For decoding the original speech audio of LJSpeech (i.e. perform resynthesis), run the following command:

```
./run.sh --stage 8 --stop-stage 8 \
    --datadir data/fixed_${N}-${K}_dedup \
    --dumpdir dump/fixed_${N}-${K}_dedup \
    --expdir exp/fixed_${N}-${K}_dedup \
    --g2p none \
    --cleaner none \
    --skip_data_prep true \
    --skip_train true \
    --inference_args "--vocoder_tag parallel_wavegan/ljspeech_style_melgan.v1" \
    --inference_tag decode_with_tacotron2 \
    --inference_nj 1 \
    --gpu_inference true
```

If you decode with VITS, specify following commands.
Note that it's convenient to set different `--inference_tag` for partitioning the result directory.

```
    --train_config conf/tuning/train_vits.yaml \
    --inference_config conf/tuning/decode_vits.yaml \
    --tts_task gan_tts \
    --inference_model train.total_count.ave_10best.pth \
    --inference_tag decode_with_vits \
```

### Speech continuation

For decoding the speech from units predicted by uLM (i.e. perform speech continuation), you first need to prepare for the continued units.
Please refer to [speechLM repository](https://github.com/mynlp/speechLM) for preparation.

Then, run `python arrange_continuation_file.py --continuation_csv_dir <continuated units file>` for generating `dump` directory.

You can synthesize continuation by following script, specifying the `dump` directory you generated.

```
./run.sh --stage 8 --stop-stage 8 \
    --g2p none \
    --cleaner none \
    --skip_data_prep true \
    --skip_train true \
    --train_set dummy \
    --valid_set dummy \
    --test_sets dev \
    --dumpdir dump/continuation/fixed_${N}-${K}_dedup_${temperature} \
    --expdir exp/fixed_${N}-${K}_dedup \
    --inference_args "--vocoder_tag parallel_wavegan/ljspeech_style_melgan.v1" \
    --inference_tag continuation_${temperature} \
    --inference_nj 1 \
    --gpu_inference true
```

## 5. Evaluation

### Speech Resynthesis

We calculate **bitrate**, **WER** (after ASR), **MCD**, **LogF0 RMSE**, and **UTMOS** for assessing the quality of speech resynthesis.

1. Calculate bitrate with `python calc_bitrate.py`.
1. Run `python asr.py` with appropriate options to perform ASR. For comparison, it is required to perform ASR on original audio. Please specify `--wav_dir data_orig/eval1/wavs --output_path myscripts/transcription/resynthesis/gold.txt` when running `asr.py`.
1. Run `python calc_er.py` for calculating WER.
1. Run `python calc_dsmetrics.py` with appropriate options for calculating MCD, LogF0 RMSE, and UTMOS.
1. Run `python plot_resynthesis_score.py` for summarizing results and visualization.

### Speech Continuation

We calculate **PPL**, **VERT**, **MMOS**, and **LLM-as-a-Judge**.
Here we explain the procedure for calculating metrics except for MMOS because it involves human evaluation.

#### Preparation

1. Run `python cut_10s.py` for clipping audio into 10 seconds of audio.
1. Run `python asr.py` with appropriate options to perform ASR on continued audio.

#### PPL and VERT

1. Run `python asr_eval_ppl.py` and `python asr_eval_bleu.py` for calculating PPL and VERT. You need to perform it on gold transcriptions for specifying the best temperature and visualization. Run `arrange_gold_transcription.py` beforehand.
1. Run `python summarize_continuation_result.py` for generating summarized csv on PPL and VERT.
1. Run `python plot_continuation_score.py` for visualization of PPL and VERT.

#### LLM-as-a-Judge

TBD

## Citing

TBD