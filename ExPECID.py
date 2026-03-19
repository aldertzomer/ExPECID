#!/usr/bin/env python3

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd


def run_cmd(cmd):
    """Run a command and exit if it fails."""
    print("Running:", shlex.join([str(x) for x in cmd]))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"Error: command failed with exit code {result.returncode}")


def check_files_exist(files):
    for f in files:
        if not Path(f).exists():
            sys.exit(f"Error: required file not found: {f}")


def build_feature_list(genelist_path):
    with open(genelist_path) as f:
        genes = [line.strip() for line in f if line.strip()]

    features = []
    for g in genes:
        features.append(f"{g}_Identity")
        features.append(f"{g}_Coverage")
    return genes, features


def infer_sample_name_from_file(path):
    """Infer a clean sample name from one input file."""
    name = Path(path).name

    suffix_patterns = [
        r"\.fastq\.gz$",
        r"\.fq\.gz$",
        r"\.fastq$",
        r"\.fq$",
        r"\.fasta$",
        r"\.fna$",
        r"\.fa$",
    ]
    for pat in suffix_patterns:
        name = re.sub(pat, "", name, flags=re.IGNORECASE)

    pair_patterns = [
        r"([._-]R?1)$",
        r"([._-]R?2)$",
    ]
    for pat in pair_patterns:
        name = re.sub(pat, "", name, flags=re.IGNORECASE)

    return name


def infer_sample_name_from_pair(r1, r2):
    """Infer sample name from paired-end files."""
    n1 = infer_sample_name_from_file(r1)
    n2 = infer_sample_name_from_file(r2)

    if n1 == n2:
        return n1

    common = os.path.commonprefix([Path(r1).name, Path(r2).name])
    common = re.sub(r"([._-]?R?)?$", "", common, flags=re.IGNORECASE)
    common = re.sub(r"[._-]+$", "", common)
    return common if common else n1


def parse_res_file(res_file, genes):
    """
    Parse KMA .res and return a dict with:
    gene_Identity, gene_Coverage
    """
    best_hits = {gene: None for gene in genes}

    with open(res_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = re.split(r"\s+", line)
            if len(parts) < 6:
                continue

            template = parts[0]
            gene = template.split(".")[0]

            if gene not in best_hits:
                continue

            try:
                score = float(parts[1])
                identity = parts[4].strip()
                coverage = parts[5].strip()
            except (ValueError, IndexError):
                continue

            current = best_hits[gene]
            if current is None or score > current["score"]:
                best_hits[gene] = {
                    "score": score,
                    "identity": identity,
                    "coverage": coverage,
                }

    row = {}
    for gene in genes:
        hit = best_hits[gene]
        if hit is None:
            row[f"{gene}_Identity"] = 0.0
            row[f"{gene}_Coverage"] = 0.0
        else:
            row[f"{gene}_Identity"] = float(hit["identity"])
            row[f"{gene}_Coverage"] = float(hit["coverage"])

    return row


def run_kma_for_sample(args, out_prefix, database_prefix, threads):
    """Run KMA for one sample."""
    if args.inputfasta:
        cmd = [
            "kma",
            "-asm",
            "-t", str(threads),
            "-nc",
            "-na",
            "-nf",
            "-i", str(args.inputfasta),
            "-t_db", str(database_prefix),
            "-o", str(out_prefix),
        ]
    elif args.inputsinglereads:
        cmd = [
            "kma",
            "-ID", "70",
            "-t", str(threads),
            "-ont",
            "-bcNano",
            "-nc",
            "-na",
            "-nf",
            "-i", str(args.inputsinglereads),
            "-t_db", str(database_prefix),
            "-o", str(out_prefix),
        ]
    elif args.inputpairedreads:
        r1, r2 = args.inputpairedreads
        cmd = [
            "kma",
            "-ID", "70",
            "-1t1",
            "-t", str(threads),
            "-nc",
            "-na",
            "-nf",
            "-ipe", str(r1), str(r2),
            "-t_db", str(database_prefix),
            "-o", str(out_prefix),
        ]
    else:
        sys.exit("Error: no valid input mode selected")

    run_cmd(cmd)


def append_predictions(output_file, pred_df):
    """Append predictions to output_file, writing header only if file does not yet exist."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_file.exists()
    pred_df.to_csv(
        output_file,
        sep="\t",
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False,
        float_format="%.2f",
    )


def validate_args(args):
    provided = sum([
        args.inputfasta is not None,
        args.inputsinglereads is not None,
        args.inputpairedreads is not None,
    ])

    if provided != 1:
        sys.exit(
            "Error: provide exactly one of -i/--inputfasta, "
            "-is/--inputsinglereads, or -ipe/--inputpairedreads"
        )

    if args.threads < 1:
        sys.exit("Error: --threads must be >= 1")

    if args.inputfasta and not args.inputfasta.exists():
        sys.exit(f"Error: input fasta not found: {args.inputfasta}")

    if args.inputsinglereads and not args.inputsinglereads.exists():
        sys.exit(f"Error: input reads not found: {args.inputsinglereads}")

    if args.inputpairedreads:
        r1, r2 = args.inputpairedreads
        if not r1.exists():
            sys.exit(f"Error: paired read 1 not found: {r1}")
        if not r2.exists():
            sys.exit(f"Error: paired read 2 not found: {r2}")


def choose_model(scriptlocation, args):
    """Choose model based on input type."""
    if args.inputfasta:
        return scriptlocation / "RFC_expec_nonexpec_full_asm_v1.0.joblib"
    return scriptlocation / "RFC_expec_nonexpec_full_reads_v1.0.joblib"


def make_workdir(sample_name, output_file, keep_temp):
    """
    Return working directory and a boolean indicating whether it is temporary.
    If keep_temp is enabled, place intermediates in:
      <output_dir>/<sample_name>_expecid_temp/
    otherwise use TemporaryDirectory.
    """
    output_file = Path(output_file)
    if keep_temp:
        workdir = output_file.parent / f"{sample_name}_expecid_temp"
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir, None
    else:
        tmp = tempfile.TemporaryDirectory(prefix=f"expecid_{sample_name}_")
        return Path(tmp.name), tmp


def main():
    parser = argparse.ArgumentParser(
        description="Score a single sample for ExPEC likelihood using KMA + Random Forest"
    )

    parser.add_argument(
        "-i", "--inputfasta",
        type=Path,
        help="Input assembly file (.fasta, .fna, .fa)"
    )
    parser.add_argument(
        "-is", "--inputsinglereads",
        type=Path,
        help="Input single-end reads (.fastq or .fastq.gz)"
    )
    parser.add_argument(
        "-ipe", "--inputpairedreads",
        nargs=2,
        type=Path,
        metavar=("R1", "R2"),
        help="Input paired-end reads (R1 R2; .fastq or .fastq.gz)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output predictions file; append if it exists, create if it does not"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of KMA threads to use (default: 1)"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep KMA intermediate files in a per-sample temp folder next to the output file"
    )

    args = parser.parse_args()
    validate_args(args)

    scriptlocation = Path(os.path.abspath(sys.argv[0])).parent
    database_comp_b = scriptlocation / "markers.comp.b"
    database_prefix = scriptlocation / "markers"
    genelist = scriptlocation / "genelist.txt"
    model_f = choose_model(scriptlocation, args)

    check_files_exist([database_comp_b, genelist, model_f])

    if args.inputfasta:
        sample_name = infer_sample_name_from_file(args.inputfasta)
    elif args.inputsinglereads:
        sample_name = infer_sample_name_from_file(args.inputsinglereads)
    else:
        sample_name = infer_sample_name_from_pair(*args.inputpairedreads)

    genes, h_format = build_feature_list(genelist)

    workdir, tmp_handle = make_workdir(sample_name, args.output, args.keep_temp)
    try:
        out_prefix = workdir / sample_name

        # Run KMA
        run_kma_for_sample(args, out_prefix, database_prefix, args.threads)

        res_file = Path(str(out_prefix) + ".res")
        if not res_file.exists():
            sys.exit(f"Error: expected KMA result file not found: {res_file}")

        # Build one-row feature table directly from .res
        feature_row = parse_res_file(res_file, genes)

        df = pd.DataFrame([{"File": sample_name, **feature_row}])
        df = df[["File"] + h_format]

        # Load model and predict
        lmodel = joblib.load(model_f)

        isolates = df["File"]
        X = df.drop(columns=["File"])

        pred = lmodel.predict(X)
        pred_prob = lmodel.predict_proba(X)

        preddf = pd.DataFrame({
            "isolate": isolates,
            "p_nonexpec": pred_prob[:, 0],
            "p_expec": pred_prob[:, 1],
            "prediction": pred,
        }).join(X, how="left")

        # Append or create output
        append_predictions(args.output, preddf)

        if args.keep_temp:
            print(f"Intermediate files kept in: {workdir}")

    finally:
        if tmp_handle is not None:
            tmp_handle.cleanup()

    print(f"\n✅ Prediction appended/written to: {args.output}")
    print(f"✅ Model used: {model_f.name}")


if __name__ == "__main__":
    main()
