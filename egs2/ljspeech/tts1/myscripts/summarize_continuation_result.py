import itertools

import pandas as pd
from plot_ppl_bleu import read_bleu, read_ppl

gold_ppl_csv = "csv/ppl/gold.txt"
gold_bleu_csv = "csv/bleu/gold.txt"
gold_ppl = read_ppl(gold_ppl_csv)
gold_bleu = read_bleu(gold_bleu_csv)

df = pd.read_csv("csv/resynthesis_result.csv", index_col=0)
gold_wer = float(df.loc["gold", "wer"])
gold_utmos = float(df.loc["gold", "UTMOS"])

NK_combinations = list(
    itertools.product(
        [20, 40, 80, 120, 160, 200, 240, 280], [2**i for i in range(7, 7 + 8)]
    )
)
wer_threshold = 5.0
utmos_threshold = 4.0
result = {}

for model in ["tacotron2", "vits"]:
    for N, K in NK_combinations:
        setting = f"{model}-{N}-{K}"
        # get data from df where index == setting and col == cer
        bitrate = int(df.loc[setting, "bitrate"])
        wer = float(df.loc[setting, "wer"])
        utmos = float(df.loc[setting, "UTMOS"])
        if wer > wer_threshold or utmos < utmos_threshold:
            result[setting] = {
                "bitrate": bitrate,
                "wer": wer,
                "utmos": utmos,
                "temperature": "",
                "PPL": "",
                "VERT": "",
                "distance": "",
            }
            continue
        # read the tsv file
        if model == "tacotron2":
            continuation_score_file = f"csv/continuation_score/{N}-{K}.tsv"
        else:
            continuation_score_file = f"csv/continuation_score/vits/{N}-{K}.tsv"
        with open(continuation_score_file) as f:
            # example content:
            # temperature	PPL	VERT	distance
            # 0.6	209.967	25.74	4.84e-01
            for line in f:
                if line.startswith("temperature"):
                    continue
                temperature, PPL, VERT, distance = line.strip().split("\t")
                result[setting] = {
                    "bitrate": bitrate,
                    "wer": float(wer),
                    "utmos": float(utmos),
                    "temperature": float(temperature),
                    "PPL": float(PPL),
                    "VERT": float(VERT),
                    "distance": float(distance),
                }
    # save result to tsv file
    with open("csv/continuation_result.csv", "w") as f:  # CHANGE
        f.write("setting,bitrate,wer,utmos,temperature,PPL,VERT,distance\n")
        f.write(f"gold,,{gold_wer},{gold_utmos},,{gold_ppl},{gold_bleu},0.0\n")
        for setting, values in result.items():
            f.write(
                f"{setting},{values['bitrate']},{values['wer']},{values['utmos']},{values['temperature']},{values['PPL']},{values['VERT']},{values['distance']}\n"
            )
