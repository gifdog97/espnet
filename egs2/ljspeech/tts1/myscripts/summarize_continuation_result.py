import pandas as pd
from plot_ppl_bleu import read_bleu, read_ppl

gold_ppl_csv = "csv/ppl/gold.txt"
gold_bleu_csv = "csv/bleu/gold.txt"
gold_ppl = read_ppl(gold_ppl_csv)
gold_bleu = read_bleu(gold_bleu_csv)

for model in ["tacotron2", "vits"]:
    result = {}
    if model == "tacotron2":
        df = pd.read_csv("csv/resynthesis_result.csv", index_col=0)
    else:
        df = pd.read_csv("csv/resynthesis_result-vits.csv", index_col=0)
    for MS in [20, 40, 80, 120, 160, 200, 240, 280]:
        for i in range(7, 15):
            K = 2**i
            setting = f"{MS}-{K}"
            # get data from df where index == setting and col == cer
            bitrate = int(df.loc[f"fixed_{setting}_dedup", "bitrate"])
            cer = float(df.loc[f"fixed_{setting}_dedup", "cer"])
            utmos = float(df.loc[f"fixed_{setting}_dedup", "utmos"])
            if cer > 20 or utmos < 4.0:
                result[setting] = {
                    "bitrate": bitrate,
                    "cer": cer,
                    "utmos": utmos,
                    "temperature": None,
                    "PPL": None,
                    "VERT": None,
                    "distance": None,
                }
                continue
            # read the tsv file
            if model == "tacotron2":
                continuation_score_file = f"csv/continuation_score/{MS}-{K}.tsv"
            else:
                continuation_score_file = f"csv/continuation_score/vits/{MS}-{K}.tsv"
            with open(continuation_score_file) as f:
                for line in f:
                    if line.startswith("temperature"):
                        continue
                    temperature, PPL, VERT, distance = line.strip().split("\t")
                    result[setting] = {
                        "bitrate": bitrate,
                        "cer": float(cer),
                        "utmos": float(utmos),
                        "temperature": float(temperature),
                        "PPL": float(PPL),
                        "VERT": float(VERT),
                        "distance": float(distance),
                    }
    # save result to tsv file
    if model == "tacotron2":
        summary_file = "csv/continuation_result.csv"
    else:
        summary_file = "csv/continuation_result-vits.csv"
    with open(summary_file, "w") as f:  # CHANGE
        f.write("setting,bitrate,cer,utmos,temperature,PPL,VERT,distance\n")
        for setting, values in result.items():
            f.write(
                f"{setting},{values['bitrate']},{values['cer']},{values['utmos']},{values['temperature']},{values['PPL']},{values['VERT']},{values['distance']}\n"
            )
        f.write(f"gold,None,None,None,None,{gold_ppl},{gold_bleu},0.0\n")
