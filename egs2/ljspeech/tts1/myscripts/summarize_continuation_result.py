import pandas as pd
from plot_ppl_bleu import read_bleu, read_ppl

# df = pd.read_csv("csv/asr-utmos.csv", index_col=0)
df = pd.read_csv("csv/asr-utmos.csv", index_col=0)  # CHANGE

result = {}
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
        with open(f"csv/continuation_score/{MS}-{K}.tsv") as f:  # CHANGE
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
gold_ppl_csv = "csv/ppl/gold.txt"
gold_bleu_csv = "csv/bleu/gold.txt"
gold_ppl = read_ppl(gold_ppl_csv)
gold_bleu = read_bleu(gold_bleu_csv)
# save result to tsv file
with open("csv/continuation_result.tsv", "w") as f:  # CHANGE
    f.write("setting\tbitrate\tcer\tutmos\ttemperature\tPPL\tVERT\tdistance\n")
    for setting, values in result.items():
        f.write(
            f"{setting}\t{values['bitrate']}\t{values['cer']}\t{values['utmos']}\t{values['temperature']}\t{values['PPL']}\t{values['VERT']}\t{values['distance']}\n"
        )
    f.write(f"gold\tNone\tNone\tNone\tNone\t{gold_ppl}\t{gold_bleu}\t0.0\n")
