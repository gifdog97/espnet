# plot distribution of perplexity values
import argparse

import matplotlib.pyplot as plt


def plot_perplexity_distribution(perplexity_values, output_file):
    plt.figure(figsize=(10, 6))
    plt.hist(perplexity_values, bins=30, color="blue", alpha=0.7)
    plt.title("Distribution of Perplexity Values")
    plt.xlabel("Perplexity")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.savefig(output_file, dpi=300)


def get_args():
    parser = argparse.ArgumentParser(
        description="Plot distribution of perplexity values"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input file containing perplexity values",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    perplexity_values = []
    with open(args.input_file, "r") as f:
        for line in f:
            if line.startswith("Average"):
                continue
            parts = line.strip().split("|")
            if "None" in parts[1]:
                continue
            perplexity_value = float(parts[1])
            perplexity_values.append(perplexity_value)
        plot_perplexity_distribution(perplexity_values, args.output_file)


if __name__ == "__main__":
    main()
