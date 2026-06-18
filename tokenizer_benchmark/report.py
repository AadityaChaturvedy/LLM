import argparse
import json
import os


def format_float(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return value if value is not None else ""


def load_results(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("results", payload)


def markdown_table(rows):
    headers = ["Dataset", "Tokenizer", "Fertility", "Chars/Tok", "UNK%", "Coverage%", "Lines/s"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = [
            row["dataset"],
            row["tokenizer"],
            format_float(row["fertility"]),
            format_float(row["chars_per_token"]),
            format_float(row["unk_rate"]),
            format_float(row["char_coverage"]),
            format_float(row["speed_lines_per_sec"]),
        ]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines) + "\n"


def latex_table(rows):
    header = (
        "\\begin{tabular}{llrrrrr}\n"
        "\\toprule\n"
        "Dataset & Tokenizer & Fertility & Chars/Tok & UNK\\% & Coverage\\% & Lines/s \\\\\n"
        "\\midrule\n"
    )
    body = []
    for row in rows:
        body.append(
            f"{row['dataset']} & {row['tokenizer']} & "
            f"{format_float(row['fertility'])} & {format_float(row['chars_per_token'])} & "
            f"{format_float(row['unk_rate'])} & {format_float(row['char_coverage'])} & "
            f"{format_float(row['speed_lines_per_sec'])} \\\\"
        )
    footer = "\n\\bottomrule\n\\end{tabular}\n"
    return header + "\n".join(body) + footer


def main():
    parser = argparse.ArgumentParser(description="Generate paper-ready tables from benchmark JSON.")
    parser.add_argument("--input", default="tokenizer_benchmark/results/main_results.json")
    parser.add_argument("--output-dir", default="tokenizer_benchmark/results")
    parser.add_argument("--sort", choices=["dataset", "fertility", "chars_per_token", "speed"], default="dataset")
    args = parser.parse_args()

    rows = load_results(args.input)
    if args.sort == "fertility":
        rows = sorted(rows, key=lambda row: (row["dataset"], row["fertility"]))
    elif args.sort == "chars_per_token":
        rows = sorted(rows, key=lambda row: (row["dataset"], -row["chars_per_token"]))
    elif args.sort == "speed":
        rows = sorted(rows, key=lambda row: (row["dataset"], -row["speed_lines_per_sec"]))
    else:
        rows = sorted(rows, key=lambda row: (row["dataset"], row["tokenizer"]))

    os.makedirs(args.output_dir, exist_ok=True)
    md_path = os.path.join(args.output_dir, "paper_table.md")
    tex_path = os.path.join(args.output_dir, "paper_table.tex")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(markdown_table(rows))
    with open(tex_path, "w", encoding="utf-8") as handle:
        handle.write(latex_table(rows))
    print(f"[DONE] Wrote {md_path}")
    print(f"[DONE] Wrote {tex_path}")


if __name__ == "__main__":
    main()
