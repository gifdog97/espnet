import itertools

import pandas as pd
from myutils import is_valid_setting
from plot_ppl_bleu import read_bleu, read_ppl


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resynthesi_result_file",
        type=str,
        help="Path to the resynthesis result CSV file.",
        default="csv/resynthesis_result.csv",
    )
    parser.add_argument(
        "--continuation_dir",
        type=str,
        help="Path to the continuation score directory.",
        default="csv/continuation_score_10s",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to output scores.",
        default="csv/continuation_result_10s.csv",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    gold_ppl_csv = "csv/ppl/gold.txt"
    gold_bleu_csv = "csv/bleu/gold.txt"
    gold_ppl = read_ppl(gold_ppl_csv)
    gold_bleu = read_bleu(gold_bleu_csv)

    df = pd.read_csv(args.resynthesi_result_file, index_col=0)
    gold_wer = float(df.loc["gold", "wer"])
    gold_utmos = float(df.loc["gold", "UTMOS"])

    NK_combinations = list(
        itertools.product(
            [20, 40, 80, 120, 160, 200, 240, 280], [2**i for i in range(7, 7 + 8)]
        )
    )
    result = {}

    for model in ["tacotron2", "vits"]:
        for N, K in NK_combinations:
            setting = f"{model}-{N}-{K}"
            # get data from df where index == setting and col == cer
            bitrate = int(df.loc[setting, "bitrate"])
            wer = float(df.loc[setting, "wer"])
            utmos = float(df.loc[setting, "UTMOS"])
            if not is_valid_setting(wer, utmos):
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
            continuation_score_file = f"{args.continuation_dir}/{model}/{N}-{K}.tsv"
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
        with open(args.output_file, "w") as f:
            f.write("setting,bitrate,wer,utmos,temperature,PPL,VERT,distance\n")
            f.write(f"gold,,{gold_wer},{gold_utmos},,{gold_ppl},{gold_bleu},0.0\n")
            for setting, values in result.items():
                f.write(
                    f"{setting},{values['bitrate']},{values['wer']},{values['utmos']},{values['temperature']},{values['PPL']},{values['VERT']},{values['distance']}\n"
                )
