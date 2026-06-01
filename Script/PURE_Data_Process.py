#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PURE: Pipeline for Uncovering Regulatory Elements (Integrated Data Processing)
Gene Regulatory Network Construction Pipeline

This comprehensive pipeline integrates evidence from co-expression analysis (GENIE3),
cross-species ChIP-seq homology transfer (via BLASTP or DIAMOND/OrthoGroups), and motif scanning
to construct high-quality, weighted Gene Regulatory Networks (GRNs).

Author: CS Li
Date: 2026-05-26
Version: 0.0.6-v2.1

v2.1 updates:
1. Adds --homology_vote_threshold for configurable TF homolog voting.
2. Adds --chip_species to select which ChIP reference species are used.
3. Adds --dry_run for non-destructive preflight checks of files, configs, tools, and common runtime hazards.
4. Adds --out_dir for clean, centralized output management.
5. Adds selectable protein-alignment backends via --aligner {blast,diamond}.
   - --aligner blast uses makeblastdb + blastp and remains the default for backward compatibility.
   - --aligner diamond uses diamond makedb + diamond blastp and writes the same three-column output schema: qseqid, sseqid, pident.
6. Adds generic alignment aliases --alignment_result_file and --alignment_evalue while retaining --blast_result_file, --blast_evalue, and --blast_evalue_cutoff for compatibility.
7. Keeps default behavior backward-compatible: threshold=0.5, aligner=blast, outputs go to the current path unless --out_dir is provided, and all ChIP species are used unless explicitly filtered.
"""

import argparse
import os
import sys
import subprocess
import logging
import tempfile
import shutil
import re
import importlib.util
import multiprocessing
from functools import partial
from collections import defaultdict
import pandas as pd
import numpy as np

# =============================================================================
# EMBEDDED R SCRIPT (GENIE3)
# =============================================================================

GENIE3_R_SCRIPT_CONTENT = r"""
#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(GENIE3))
suppressPackageStartupMessages(library(data.table))

normalize_weights <- function(weights) {
  min_w <- min(weights)
  max_w <- max(weights)
  if (max_w == min_w) { return(rep(100, length(weights))) }
  return(1 + 99 * (weights - min_w) / (max_w - min_w))
}

parser <- ArgumentParser(description="Run GENIE3")
parser$add_argument("--rna_matrix", required=TRUE)
parser$add_argument("--tf_list", required=TRUE)
parser$add_argument("--output", required=TRUE)
parser$add_argument("--threads", type="integer", default=1)
parser$add_argument("--filter_method", default="q20")
args <- parser$parse_args()

cat(sprintf("--- Loading RNA matrix: %s ---\n", args$rna_matrix))
expr_matrix <- as.matrix(fread(args$rna_matrix, data.table=TRUE), rownames=1)

tf_list <- fread(args$tf_list, header=FALSE, col.names=c("geneID", "Family"))
tf_names <- intersect(tf_list$geneID, rownames(expr_matrix))

if (length(tf_names) == 0) { stop("Fatal: No TFs found in RNA matrix.") }
cat(sprintf("Identified %d TFs present in expression matrix.\n", length(tf_names)))

expr_matrix_log <- log2(expr_matrix + 1)
cat(sprintf("Running GENIE3 with %d threads...\n", args$threads))
weight_matrix <- GENIE3(expr_matrix_log, regulators = tf_names, nCores = args$threads)
link_list <- getLinkList(weight_matrix)
setDT(link_list)

raw_output_file <- sub("\\.tsv$", "_raw_links.csv", args$output)
fwrite(link_list, file = raw_output_file, sep = ",", col.names = TRUE)

cat(sprintf("Filtering results: %s\n", args$filter_method))
if (grepl("^q", args$filter_method)) {
  q_val <- as.numeric(sub("q", "", args$filter_method)) / 100
  threshold <- quantile(link_list$weight, 1 - q_val)
  filtered_list <- link_list[weight >= threshold]
} else if (grepl("k$", args$filter_method)) {
  top_n <- as.numeric(sub("k", "", args$filter_method)) * 1000
  setorder(link_list, -weight)
  filtered_list <- head(link_list, n=top_n)
} else { stop("Unknown filter method.") }

filtered_list[, norm_weight := normalize_weights(weight)]
output_data <- filtered_list[, .(regulatoryGene, targetGene, norm_weight)]
fwrite(output_data, file = args$output, sep = "\t", col.names = FALSE)
cat("--- GENIE3 Completed ---\n")
"""

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Global variable to hold the log file path
LOG_FILE_PATH = None

def setup_logging(log_file):
    """Configures logging to a single file and console."""
    global LOG_FILE_PATH
    LOG_FILE_PATH = log_file

    # Remove existing handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def validate_input_files(args):
    """Pre-flight check to ensure all input files exist."""
    files_to_check = {
        "--target_genome_config": args.target_genome_config,
        "--target_tf_list": args.target_tf_list,
        "--chip_ref_genome_config": args.chip_ref_genome_config,
        "--chip_peak_config": args.chip_peak_config,
        "--atac_peak_config": args.atac_peak_config,
        "--motif_list": args.motif_list,
        "--motif_file": args.motif_file
    }

    # Optional files
    if args.rna_matrix: files_to_check["--rna_matrix"] = args.rna_matrix
    if args.genie3_file: files_to_check["--genie3_file"] = args.genie3_file
    if args.blast_result_file: files_to_check["--blast_result_file"] = args.blast_result_file

    missing = []
    for arg_name, file_path in files_to_check.items():
        if file_path and not os.path.exists(file_path):
            missing.append(f"{arg_name}: {file_path}")

    if missing:
        logging.error("FATAL: The following input files were not found:")
        for m in missing:
            logging.error(f"  - {m}")
        logging.error("Please check your paths and try again.")
        sys.exit(1)

def run_command(cmd, shell=False, check=True, capture_output_file=None):
    """
    Executes a subprocess command.

    Behavior:
    1. Logs the command.
    2. If capture_output_file is provided, stdout is written there (stderr to main log).
    3. If capture_output_file is None, BOTH stdout and stderr are appended to the main log file.
    """
    cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
    logging.info(f"CMD: {cmd_str}")

    # Determine where to send output
    # If specific file requested for data capture (e.g. bedtools intersect > file), use it for stdout
    # stderr always goes to the main log to keep it "clean" but comprehensive

    try:
        with open(LOG_FILE_PATH, 'a') as main_log:
            if capture_output_file:
                # Stdout to file, Stderr to main log
                subprocess.run(cmd, shell=shell, check=check, stdout=capture_output_file, stderr=main_log, text=True)
            else:
                # Everything to main log
                subprocess.run(cmd, shell=shell, check=check, stdout=main_log, stderr=main_log, text=True)

    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}: {cmd_str}")
        logging.error("Check the log file above for specific error messages from the tool.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Execution error: {e}")
        sys.exit(1)

def load_config_map(config_path, key_idx=0, val_idx=1):
    """Loads a TSV config into a dictionary."""
    mapping = {}
    with open(config_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > max(key_idx, val_idx):
                mapping[parts[key_idx]] = parts[val_idx]
    return mapping


def parse_species_filter(species_arg):
    """
    Parses a comma-separated ChIP reference species selector.

    Parameters
    ----------
    species_arg : str or None
        Examples: "A_thaliana,Z_mays,O_sativa", "all", None.

    Returns
    -------
    set or None
        None means "use every species present in the ChIP reference config",
        preserving the historical behavior of the pipeline.
    """
    if not species_arg:
        return None

    raw = species_arg.strip()
    if raw.lower() in {"all", "*"}:
        return None

    species = {item.strip() for item in raw.split(',') if item.strip()}
    return species if species else None


def parse_target_species_prefix(target_tf_list):
    """
    Infers the target gene-ID prefix from the first TF ID.

    This preserves the original prefix heuristic while centralizing it for
    validation, dry-run reporting, and orthology alignment.
    """
    with open(target_tf_list) as f:
        first_tf = f.readline().split('\t')[0].strip()
        m = re.match(r'^([^0-9]+)', first_tf)
        if not m:
            raise ValueError("Cannot determine species prefix from TF list.")
        return m.group(1)


def configure_output_paths(args, create_dir=False):
    """
    Resolves and normalizes all pipeline output locations.

    Historical behavior is preserved when --out_dir is not provided: --out_prefix
    may be a bare prefix or a path-like prefix. When --out_dir is provided, all
    pipeline-generated files are written into that directory and the basename of
    --out_prefix is used as the output file prefix.

    The resolved absolute prefix is stored back in args.out_prefix so existing
    downstream code can keep constructing outputs with f"{args.out_prefix}_...".
    The original user-provided value is retained in args.raw_out_prefix.
    """
    raw_prefix = args.out_prefix
    if not raw_prefix or not raw_prefix.strip():
        raise ValueError("--out_prefix must not be empty.")

    args.raw_out_prefix = raw_prefix

    if args.out_dir:
        output_dir = os.path.abspath(os.path.expanduser(args.out_dir))
        prefix_name = os.path.basename(os.path.normpath(raw_prefix))
        if not prefix_name:
            raise ValueError("--out_prefix must include a valid file-name prefix.")
        output_prefix = os.path.join(output_dir, prefix_name)
    else:
        output_prefix = os.path.abspath(os.path.expanduser(raw_prefix))
        output_dir = os.path.dirname(output_prefix) or os.getcwd()
        prefix_name = os.path.basename(output_prefix)

    if create_dir:
        os.makedirs(output_dir, exist_ok=True)

    args.out_dir = output_dir
    args.out_prefix_name = prefix_name
    args.out_prefix = output_prefix
    return output_prefix


def prefixed_output(out_prefix, suffix):
    """Returns an output path by appending a pipeline suffix to a prefix path."""
    return f"{out_prefix}{suffix}"


def read_config_records(config_path, min_cols=1, label="config"):
    """
    Reads a tab-delimited config file for validation.

    Blank lines and comment lines are ignored. The function returns both the
    parsed records and a list of human-readable format errors so callers can
    aggregate all dry-run findings instead of failing at the first issue.
    """
    records = []
    errors = []

    if not config_path or not os.path.exists(config_path):
        return records, [f"{label} not found: {config_path}"]

    with open(config_path, 'r') as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip('\n')
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < min_cols:
                errors.append(
                    f"{label}:{line_no} expects >= {min_cols} columns, observed {len(parts)}: {line}"
                )
                continue
            records.append(parts)

    if not records:
        errors.append(f"{label} contains no usable records: {config_path}")

    return records, errors


def check_file_exists(path, label, errors, warnings=None, required=True):
    """Adds a dry-run finding if a file path is missing, empty, or suspicious."""
    if not path:
        if required:
            errors.append(f"{label} is required but was not provided.")
        return False

    if not os.path.exists(path):
        (errors if required else warnings).append(f"{label} not found: {path}")
        return False

    if os.path.isdir(path):
        errors.append(f"{label} points to a directory, expected a file: {path}")
        return False

    try:
        if os.path.getsize(path) == 0:
            (errors if required else warnings).append(f"{label} is empty: {path}")
    except OSError as exc:
        warnings.append(f"Could not inspect {label}: {path} ({exc})")

    return True


def check_command_available(command, errors, warnings=None, required=True):
    """Checks whether an external executable is available on PATH."""
    if shutil.which(command):
        return True

    message = f"Required executable not found on PATH: {command}"
    (errors if required else warnings).append(message)
    return False


def check_python_module(module_name, errors, warnings=None, required=True):
    """Checks whether a Python module required at runtime is importable."""
    if importlib.util.find_spec(module_name) is not None:
        return True

    message = f"Python module not importable: {module_name}"
    (errors if required else warnings).append(message)
    return False


def check_output_writable(out_prefix, errors, warnings):
    """Checks whether the output directory appears writable and warns on overwrite."""
    out_dir = os.path.dirname(os.path.abspath(out_prefix)) or os.getcwd()
    if not os.path.exists(out_dir):
        errors.append(f"Output directory does not exist: {out_dir}")
        return

    if not os.access(out_dir, os.W_OK):
        errors.append(f"Output directory is not writable: {out_dir}")

    predictable_outputs = [
        prefixed_output(out_prefix, ".log"),
        prefixed_output(out_prefix, "_GENIE3_calculated.tsv"),
        prefixed_output(out_prefix, "_ChIP_ref_targets.tsv"),
        prefixed_output(out_prefix, "_blast_generated.tsv"),
        prefixed_output(out_prefix, "_diamond_generated.tsv"),
        prefixed_output(out_prefix, "_pseudoChIP_targets_raw.tsv"),
        prefixed_output(out_prefix, "_GENIE3_motif_filtered.tsv"),
        prefixed_output(out_prefix, "_pseudoChIP_targets_motif_filtered.tsv"),
    ]
    for output_path in predictable_outputs:
        # The current log file is intentionally opened before dry-run reporting;
        # do not report it as an overwrite hazard for the current invocation.
        if LOG_FILE_PATH and os.path.abspath(output_path) == os.path.abspath(LOG_FILE_PATH):
            continue
        if os.path.exists(output_path):
            warnings.append(f"Existing output may be overwritten: {output_path}")


def check_tmp_dir(tmp_dir, errors, warnings):
    """Checks custom temporary directory safety."""
    if not tmp_dir:
        return

    tmp_dir = os.path.abspath(tmp_dir)
    if os.path.exists(tmp_dir):
        if not os.path.isdir(tmp_dir):
            errors.append(f"--tmp_dir exists but is not a directory: {tmp_dir}")
        elif not os.access(tmp_dir, os.W_OK):
            errors.append(f"--tmp_dir is not writable: {tmp_dir}")
    else:
        parent = os.path.dirname(tmp_dir) or os.getcwd()
        if not os.path.exists(parent):
            errors.append(f"Parent directory for --tmp_dir does not exist: {parent}")
        elif not os.access(parent, os.W_OK):
            errors.append(f"Parent directory for --tmp_dir is not writable: {parent}")


def check_r_packages(errors, warnings):
    """
    Performs a lightweight R package check used only in --dry_run mode.

    The command is intentionally small and does not run GENIE3; it simply
    verifies that the R packages used by the embedded script can be loaded.
    """
    if not shutil.which("Rscript"):
        return

    r_expr = (
        "pkgs <- c('argparse','GENIE3','data.table'); "
        "missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly=TRUE)]; "
        "if (length(missing)) { stop(paste(missing, collapse=',')) }"
    )
    try:
        proc = subprocess.run(
            ["Rscript", "-e", r_expr],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False
        )
        if proc.returncode != 0:
            warnings.append(
                "Rscript is available, but one or more required R packages are missing "
                f"or not loadable: argparse, GENIE3, data.table. Detail: {proc.stderr.strip()}"
            )
    except Exception as exc:
        warnings.append(f"Could not complete R package check: {exc}")


def dry_run_preflight(args, selected_chip_species):
    """
    Runs a non-destructive pre-flight audit.

    The dry run does not execute sequence alignment, bedtools intersections, FIMO, GENIE3,
    or write final network outputs. It validates file paths, config consistency,
    selected ChIP species, external executables, writable locations, and common
    runtime hazards.
    """
    errors = []
    warnings = []
    info = []

    logging.info("=== PURE dry-run preflight started ===")
    logging.info("No analysis commands will be executed in dry-run mode.")
    logging.info("Resolved output directory: %s", args.out_dir)
    logging.info("Resolved output prefix path: %s", args.out_prefix)

    # Basic CLI sanity checks.
    if args.threads < 1:
        errors.append("--threads must be >= 1.")
    elif os.cpu_count() and args.threads > os.cpu_count():
        warnings.append(
            f"--threads={args.threads} exceeds detected CPU count ({os.cpu_count()}); "
            "this may reduce performance on shared systems."
        )

    if not (0.0 <= args.homology_vote_threshold <= 1.0):
        errors.append("--homology_vote_threshold must be between 0 and 1.")

    # Top-level input files.
    top_level_files = {
        "--target_genome_config": args.target_genome_config,
        "--target_tf_list": args.target_tf_list,
        "--chip_ref_genome_config": args.chip_ref_genome_config,
        "--chip_peak_config": args.chip_peak_config,
        "--atac_peak_config": args.atac_peak_config,
        "--motif_list": args.motif_list,
        "--motif_file": args.motif_file,
    }
    for label, path in top_level_files.items():
        check_file_exists(path, label, errors, warnings)

    if args.genie3_file:
        check_file_exists(args.genie3_file, "--genie3_file", errors, warnings)
    elif args.rna_matrix:
        check_file_exists(args.rna_matrix, "--rna_matrix", errors, warnings)
    else:
        errors.append("Either --genie3_file or --rna_matrix is required.")

    if args.blast_result_file:
        check_file_exists(args.blast_result_file, "--alignment_result_file/--blast_result_file", errors, warnings)

    # Config-level validation.
    target_records, cfg_errors = read_config_records(args.target_genome_config, 4, "target genome config")
    errors.extend(cfg_errors)

    target_species = None
    if target_records:
        target_species = target_records[0][0]
        info.append(f"Target species from target genome config: {target_species}")
        if len(target_records) > 1:
            warnings.append(
                "Target genome config contains multiple records; the pipeline currently uses only the first one."
            )
        check_file_exists(target_records[0][1], f"target genome FASTA ({target_species})", errors, warnings)
        check_file_exists(target_records[0][2], f"target GFF ({target_species})", errors, warnings)
        check_file_exists(target_records[0][3], f"target peptide FASTA ({target_species})", errors, warnings)

    tf_records, cfg_errors = read_config_records(args.target_tf_list, 2, "target TF list")
    errors.extend(cfg_errors)
    if tf_records:
        info.append(f"Target TF records: {len(tf_records)}")
        try:
            prefix = parse_target_species_prefix(args.target_tf_list)
            info.append(f"Inferred target gene-ID prefix: {prefix}")
        except Exception as exc:
            errors.append(str(exc))

    ref_records, cfg_errors = read_config_records(args.chip_ref_genome_config, 4, "ChIP reference genome config")
    errors.extend(cfg_errors)
    ref_species = {r[0]: r for r in ref_records}

    peak_records, cfg_errors = read_config_records(args.chip_peak_config, 5, "ChIP peak config")
    errors.extend(cfg_errors)

    atac_records, cfg_errors = read_config_records(args.atac_peak_config, 2, "ATAC peak config")
    errors.extend(cfg_errors)
    atac_species = {r[0]: r[1] for r in atac_records}

    if selected_chip_species:
        info.append("Selected ChIP species: " + ",".join(sorted(selected_chip_species)))
        missing_ref = sorted(selected_chip_species - set(ref_species))
        if missing_ref:
            errors.append(
                "Selected ChIP species absent from --chip_ref_genome_config: "
                + ",".join(missing_ref)
            )
        active_chip_species = set(selected_chip_species)
    else:
        info.append("Selected ChIP species: all species in --chip_ref_genome_config")
        active_chip_species = set(ref_species)

    # Reference genomes and ATAC files for ChIP species.
    for species in sorted(active_chip_species & set(ref_species)):
        record = ref_species[species]
        check_file_exists(record[1], f"ChIP reference genome FASTA ({species})", errors, warnings)
        check_file_exists(record[2], f"ChIP reference GFF ({species})", errors, warnings)
        check_file_exists(record[3], f"ChIP reference peptide FASTA ({species})", errors, warnings)

        if species in atac_species:
            check_file_exists(atac_species[species], f"ATAC peaks ({species})", errors, warnings)
        else:
            msg = f"No ATAC peak entry for selected ChIP species: {species}"
            if selected_chip_species:
                errors.append(msg)
            else:
                warnings.append(msg + " (species will be skipped, matching historical behavior).")

    if target_species:
        if target_species in atac_species:
            check_file_exists(atac_species[target_species], f"target ATAC peaks ({target_species})", errors, warnings)
        else:
            errors.append(
                f"Target species {target_species} is absent from --atac_peak_config; motif filtering cannot run."
            )

    # ChIP peak paths are checked only for active species to avoid forcing unused
    # species to be valid when the user intentionally filters them out.
    peaks_by_species = defaultdict(int)
    for record in peak_records:
        species = record[0]
        if active_chip_species and species not in active_chip_species:
            continue
        peaks_by_species[species] += 1
        check_file_exists(record[4], f"ChIP peak file ({species}, {record[2]})", errors, warnings)

    for species in sorted(active_chip_species):
        if peaks_by_species.get(species, 0) == 0:
            warnings.append(f"No ChIP peak records found for active species: {species}")

    motif_records, cfg_errors = read_config_records(args.motif_list, 2, "motif-family config")
    errors.extend(cfg_errors)
    if motif_records:
        info.append(f"Motif-family records: {len(motif_records)}")

    if os.path.exists(args.motif_file):
        try:
            with open(args.motif_file) as motif_handle:
                first_line = motif_handle.readline().strip()
            if not first_line.startswith("MEME"):
                warnings.append(
                    f"Motif file does not start with a MEME header. First line: {first_line}"
                )
        except Exception as exc:
            warnings.append(f"Could not inspect motif MEME header: {exc}")

    # Weight tuple validation.
    for item in args.coexpr_chip_motif:
        parts = [p.strip() for p in item.split(',')]
        if len(parts) != 3:
            errors.append(f"Invalid --coexpr_chip_motif entry: {item}")
            continue
        try:
            w_coexpr = float(parts[0])
            w_chip = float(parts[1])
        except ValueError:
            errors.append(f"Invalid numeric weights in --coexpr_chip_motif entry: {item}")
            continue
        use_motif = parts[2].upper()
        if w_coexpr < 0 or w_chip < 0:
            errors.append(f"Weights must be non-negative in --coexpr_chip_motif entry: {item}")
        if use_motif not in {"Y", "N"}:
            errors.append(f"Motif flag must be Y or N in --coexpr_chip_motif entry: {item}")
        if w_coexpr == 0 and w_chip == 0:
            warnings.append(f"Both weights are zero; output would be empty for: {item}")

    # Runtime environment checks.
    check_python_module("tables", errors, warnings, required=False)

    check_command_available("bedtools", errors, warnings)
    check_command_available("fimo", errors, warnings)

    if not args.blast_result_file:
        if args.aligner == "blast":
            check_command_available("makeblastdb", errors, warnings)
            check_command_available("blastp", errors, warnings)
        elif args.aligner == "diamond":
            check_command_available("diamond", errors, warnings)
        else:
            errors.append(f"Unsupported --aligner value: {args.aligner}")
    else:
        info.append("Using a precomputed alignment table; --aligner will not run a new search.")

    if not args.genie3_file:
        if check_command_available("Rscript", errors, warnings):
            check_r_packages(errors, warnings)

    check_output_writable(args.out_prefix, errors, warnings)
    check_tmp_dir(args.tmp_dir, errors, warnings)

    # Report.
    for message in info:
        logging.info(f"[OK] {message}")

    if warnings:
        logging.warning("--- Dry-run warnings ---")
        for message in warnings:
            logging.warning(f"[WARN] {message}")

    if errors:
        logging.error("--- Dry-run errors ---")
        for message in errors:
            logging.error(f"[ERROR] {message}")
        logging.error("Dry-run failed. Fix the errors above before running the full pipeline.")
        return False

    logging.info("Dry-run passed. No fatal problems were detected.")
    return True


# =============================================================================
# MODULE 1: CHIP PEAK ASSOCIATION & PROCESSING
# =============================================================================

def parse_gff_tss(gff_file):
    """Parses GFF to extract TSS coordinates (strand-aware)."""
    tss_list = []
    with open(gff_file, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            cols = line.strip().split('\t')
            if len(cols) < 9 or cols[2] != 'gene': continue

            attrs = cols[8]
            gene_id = None
            if "ID=" in attrs:
                gene_id = attrs.split("ID=")[1].split(";")[0]
            elif "Name=" in attrs:
                gene_id = attrs.split("Name=")[1].split(";")[0]

            if not gene_id: continue

            chrom, start, end, strand = cols[0], int(cols[3]), int(cols[4]), cols[6]
            tss_pos = start if strand == '+' else end
            tss_list.append((chrom, tss_pos, tss_pos + 1, gene_id, "0", strand))
    return tss_list

def create_promoter_bed(tss_list, upstream=2500, downstream=100):
    """Creates promoter regions from TSS."""
    promoters = []
    for chrom, tss_start, tss_end, gene_id, score, strand in tss_list:
        if strand == '+':
            p_start = max(0, tss_start - upstream)
            p_end = tss_start + downstream
        else:
            p_start = max(0, tss_end - downstream)
            p_end = tss_end + upstream
        promoters.append(f"{chrom}\t{p_start}\t{p_end}\t{gene_id}\t{score}\t{strand}")
    return promoters

def process_chip_data(chip_ref_config, chip_peak_config, atac_peak_config, output_file, tmp_dir, selected_chip_species=None):
    logging.info("--- Starting ChIP Peak Association ---")
    if selected_chip_species:
        logging.info("Restricting ChIP reference species to: %s", ",".join(sorted(selected_chip_species)))

    ref_genomes = {}
    with open(chip_ref_config) as f:
        for line in f:
            p = line.strip().split('\t')
            ref_genomes[p[0]] = {'gff': p[2]}

    atac_files = load_config_map(atac_peak_config)
    species_gene_atac_map = {}

    processed_species = set()
    for species, genome_data in ref_genomes.items():
        if selected_chip_species and species not in selected_chip_species:
            continue
        if species not in atac_files: continue
        logging.info(f"Processing reference species: {species}")

        gff_file = genome_data['gff']
        atac_file = atac_files[species]

        tss_data = parse_gff_tss(gff_file)
        promoters = create_promoter_bed(tss_data)

        promoter_bed_path = os.path.join(tmp_dir, f"{species}_promoters.bed")
        with open(promoter_bed_path, 'w') as f:
            f.write("\n".join(promoters))

        gene_atac_bed = os.path.join(tmp_dir, f"{species}_gene_atac.bed")

        # Intersect promoter and ATAC
        cmd = ["bedtools", "intersect", "-a", promoter_bed_path, "-b", atac_file, "-wa", "-wb"]

        # We need to capture this output to process it, not just dump to log
        # So we use subprocess directly here to stream line-by-line
        try:
            with open(gene_atac_bed, 'w') as out_f:
                # Stderr goes to main log
                with open(LOG_FILE_PATH, 'a') as log_f:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log_f, text=True)

                    for line in proc.stdout:
                        cols = line.strip().split('\t')
                        if len(cols) >= 9:
                            out_f.write(f"{cols[6]}\t{cols[7]}\t{cols[8]}\t{cols[3]}\n")

                    if proc.wait() != 0:
                        raise subprocess.CalledProcessError(proc.returncode, cmd)
        except Exception as e:
            logging.error(f"Error processing ATAC intersection for {species}: {e}")
            sys.exit(1)

        species_gene_atac_map[species] = gene_atac_bed
        processed_species.add(species)

    if selected_chip_species:
        skipped = selected_chip_species - processed_species
        if skipped:
            logging.warning("Selected ChIP species could not be processed, usually because ATAC entries are missing: %s", ",".join(sorted(skipped)))

    # Process Peaks
    with open(output_file, 'w') as final_out:
        with open(chip_peak_config, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 5: continue
                species, tf_id, tf_name, family, peak_path = parts
                if selected_chip_species and species not in selected_chip_species:
                    continue

                if species not in species_gene_atac_map or not os.path.exists(peak_path):
                    continue

                logging.info(f"Processing TF: {tf_name} ({species})")

                # Filter self-binding
                tf_coords_bed = os.path.join(tmp_dir, "tf_coords.tmp")
                found_tf = False
                with open(ref_genomes[species]['gff'], 'r') as gff:
                    for gline in gff:
                        if tf_id in gline and "\tgene\t" in gline:
                            c = gline.strip().split('\t')
                            with open(tf_coords_bed, 'w') as tfo:
                                tfo.write(f"{c[0]}\t{c[3]}\t{c[4]}\n")
                            found_tf = True
                            break

                filtered_peak_path = os.path.join(tmp_dir, "peak_filtered.tmp")
                if found_tf:
                    with open(filtered_peak_path, 'w') as fp:
                        run_command(["bedtools", "intersect", "-a", peak_path, "-b", tf_coords_bed, "-v"], capture_output_file=fp)
                else:
                    shutil.copy(peak_path, filtered_peak_path)

                # Normalize Signal
                try:
                    df = pd.read_csv(filtered_peak_path, sep='\t', header=None)
                    if df.empty: continue

                    min_val = df.iloc[:, 6].min()
                    max_val = df.iloc[:, 6].max()

                    if max_val == min_val:
                        df.iloc[:, 6] = 100
                    else:
                        df.iloc[:, 6] = 1 + 99 * ((df.iloc[:, 6] - min_val) / (max_val - min_val))

                    summits = df.copy()
                    summits[1] = summits[1] + summits[9]
                    summits[2] = summits[1] + 1
                    summits = summits.iloc[:, [0, 1, 2, 6]]

                    summit_path = os.path.join(tmp_dir, "summits.bed")
                    summits.to_csv(summit_path, sep='\t', header=False, index=False)

                    # Intersect Summits with Gene-ATAC
                    # Use subprocess directly for streaming capture
                    cmd_int = ["bedtools", "intersect", "-a", summit_path, "-b", species_gene_atac_map[species], "-wa", "-wb"]

                    with open(LOG_FILE_PATH, 'a') as log_f:
                        proc = subprocess.Popen(cmd_int, stdout=subprocess.PIPE, stderr=log_f, text=True)
                        for row in proc.stdout:
                            c = row.strip().split('\t')
                            final_out.write(f"{tf_id}\t{c[7]}\t{c[3]}\n")
                        proc.wait()

                except Exception as e:
                    logging.warning(f"Error processing peaks for {tf_name}: {e}")

# =============================================================================
# MODULE 2: SEQUENCE ALIGNMENT & ORTHOLOGY TRANSFER
# =============================================================================

def collect_peptide_fastas(target_config, ref_config):
    """Collects target and reference peptide FASTA paths from genome config files."""
    pep_files = []
    with open(target_config) as handle:
        first = handle.readline().strip().split('\t')
        if len(first) >= 4:
            pep_files.append(first[3])

    with open(ref_config) as handle:
        for line in handle:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                pep_files.append(parts[3])

    return pep_files


def concatenate_peptide_fastas(pep_files, combined_fasta):
    """Writes a combined protein FASTA used as both query and search database."""
    with open(combined_fasta, 'w') as outfile:
        for peptide_fasta in pep_files:
            if not os.path.exists(peptide_fasta):
                logging.warning("Peptide FASTA not found and will be skipped: %s", peptide_fasta)
                continue
            with open(peptide_fasta) as infile:
                shutil.copyfileobj(infile, outfile)


def normalize_alignment_output(output_file):
    """Removes self-hits while preserving a three-column tabular alignment file."""
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        open(output_file, 'w').close()
        logging.warning("Alignment produced no hits: %s", output_file)
        return

    df = pd.read_csv(output_file, sep='\t', names=['q', 's', 'p'])
    df = df[df['q'] != df['s']]
    df.to_csv(output_file, sep='\t', index=False, header=False)


def run_sequence_alignment_pipeline(target_config, ref_config, output_file, threads, evalue, tmp_dir, aligner="blast"):
    """
    Runs all-vs-all protein similarity search using BLASTP or DIAMOND.

    The output schema is intentionally identical for both aligners:
    qseqid, sseqid, and percent identity. This keeps the downstream OrthoGroup
    and homolog voting logic unchanged.
    """
    aligner = aligner.lower()
    if aligner not in {"blast", "diamond"}:
        raise ValueError(f"Unsupported aligner: {aligner}")

    logging.info("--- Starting protein similarity search with %s ---", aligner.upper())

    pep_files = collect_peptide_fastas(target_config, ref_config)
    combined_fasta = os.path.join(tmp_dir, "all_prot.fasta")
    concatenate_peptide_fastas(pep_files, combined_fasta)

    if os.path.getsize(combined_fasta) == 0:
        logging.error("No peptide sequences were written to the combined FASTA. Check genome config peptide paths.")
        sys.exit(1)

    if aligner == "blast":
        db_path = os.path.join(tmp_dir, "blastdb")
        run_command(["makeblastdb", "-in", combined_fasta, "-dbtype", "prot", "-out", db_path])
        logging.info("Running BLASTP all-vs-all search...")
        run_command([
            "blastp", "-query", combined_fasta, "-db", db_path, "-out", output_file,
            "-evalue", evalue, "-num_threads", str(threads), "-outfmt", "6 qseqid sseqid pident"
        ])

    else:
        db_path = os.path.join(tmp_dir, "diamond_db")
        run_command(["diamond", "makedb", "--in", combined_fasta, "-d", db_path])
        logging.info("Running DIAMOND BLASTP-compatible all-vs-all search...")
        run_command([
            "diamond", "blastp", "-q", combined_fasta, "-d", db_path, "-o", output_file,
            "-e", evalue, "-p", str(threads), "-f", "6", "qseqid", "sseqid", "pident"
        ])

    normalize_alignment_output(output_file)


def run_blast_pipeline(target_config, ref_config, output_file, threads, evalue, tmp_dir):
    """Backward-compatible wrapper for the historical BLASTP implementation."""
    run_sequence_alignment_pipeline(
        target_config=target_config,
        ref_config=ref_config,
        output_file=output_file,
        threads=threads,
        evalue=evalue,
        tmp_dir=tmp_dir,
        aligner="blast",
    )

def build_orthogroups(alignment_file, relevant_tfs, tmp_dir):
    logging.info("Building TF OrthoGroups...")
    adj = defaultdict(set)
    relevant_set = set(relevant_tfs)

    with open(alignment_file, 'r') as f:
        for line in f:
            q, s, _ = line.strip().split('\t')
            if q in relevant_set and s in relevant_set:
                adj[q].add(s)
                adj[s].add(q)

    visited = set()
    groups = []
    for node in relevant_set:
        if node not in visited and node in adj:
            component = []
            queue = [node]
            visited.add(node)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            groups.append(component)
    return groups

def process_homology_voting_chunk(chunk_targets, groups, alignment_file, chip_data_file, target_species_prefix, vote_threshold=0.5):
    homologs = defaultdict(set)
    with open(alignment_file, 'r') as f:
        for line in f:
            q, s, _ = line.strip().split('\t')
            homologs[q].add(s)
            homologs[s].add(q)

    tf_signals = defaultdict(dict)
    tfs_with_data = set()
    with open(chip_data_file, 'r') as f:
        for line in f:
            c = line.strip().split('\t')
            if len(c) < 3: continue
            tf, target, sig = c
            try: sig = float(sig)
            except: continue
            tf_signals[tf][target] = sig
            tfs_with_data.add(tf)

    results = []

    for target_gene in chunk_targets:
        ref_homologs = homologs.get(target_gene, set())
        species_homologs = defaultdict(set)
        for h in ref_homologs:
            import re
            m = re.match(r'^([^0-9]+)', h)
            if m: species_homologs[m.group(1)].add(h)
        m_tgt = re.match(r'^([^0-9]+)', target_gene)
        if m_tgt: species_homologs[m_tgt.group(1)].add(target_gene)

        for ref_sp, ref_genes in species_homologs.items():
            if not ref_genes: continue
            for g_idx, members in enumerate(groups):
                members_in_sp = [m for m in members if m.startswith(ref_sp)]
                if not members_in_sp: continue

                t_c = 0
                v_c = 0
                local_max_sig = 0
                for tf in members_in_sp:
                    if tf in tfs_with_data:
                        t_c += 1
                        binding_strength = 0
                        is_binding = False
                        if tf in tf_signals:
                            for rg in ref_genes:
                                if rg in tf_signals[tf]:
                                    s = tf_signals[tf][rg]
                                    if s > binding_strength: binding_strength = s
                                    is_binding = True
                        if is_binding:
                            v_c += 1
                            if binding_strength > local_max_sig:
                                local_max_sig = binding_strength
                if t_c > 0 and (v_c / t_c) >= vote_threshold:
                    for member in members:
                        if member.startswith(target_species_prefix):
                            results.append((member, target_gene, local_max_sig))
    return results

def align_orthogroups(chip_target_file, alignment_file, chip_peak_config, target_tf_list, output_file, threads, tmp_dir, selected_chip_species=None, vote_threshold=0.5):
    logging.info("--- Starting OrthoGroup alignment and homolog voting ---")
    logging.info("Homology voting threshold: %.3f", vote_threshold)
    if selected_chip_species:
        logging.info("ChIP species used for orthology voting: %s", ",".join(sorted(selected_chip_species)))

    with open(target_tf_list) as f:
        first_tf = f.readline().split('\t')[0]
        m = re.match(r'^([^0-9]+)', first_tf)
        if not m: raise ValueError("Cannot determine species prefix from TF list")
        target_prefix = m.group(1)

    tfs = set()
    with open(target_tf_list) as f:
        for line in f: tfs.add(line.split('\t')[0])
    with open(chip_peak_config) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            if selected_chip_species and parts[0] not in selected_chip_species:
                continue
            tfs.add(parts[1])

    groups = build_orthogroups(alignment_file, list(tfs), tmp_dir)
    logging.info(f"Generated {len(groups)} OrthoGroups.")

    target_genes = set()
    with open(alignment_file) as f:
        for line in f:
            q, s, _ = line.strip().split('\t')
            if q.startswith(target_prefix): target_genes.add(q)
            if s.startswith(target_prefix): target_genes.add(s)
    target_genes = sorted(list(target_genes))

    chunk_size = len(target_genes) // threads + 1
    chunks = [target_genes[i:i + chunk_size] for i in range(0, len(target_genes), chunk_size)]

    logging.info(f"Processing in {threads} threads...")
    worker = partial(process_homology_voting_chunk,
                     groups=groups, alignment_file=alignment_file,
                     chip_data_file=chip_target_file, target_species_prefix=target_prefix,
                     vote_threshold=vote_threshold)

    with multiprocessing.Pool(threads) as pool:
        all_results = pool.map(worker, chunks)

    logging.info("Combining results...")
    final_interactions = defaultdict(float)
    for res_list in all_results:
        for tf, tgt, sig in res_list:
            if sig > final_interactions[(tf, tgt)]:
                final_interactions[(tf, tgt)] = sig

    with open(chip_target_file) as f:
        for line in f:
            tf, tgt, sig = line.strip().split('\t')
            if tf.startswith(target_prefix) and tgt.startswith(target_prefix):
                s = float(sig)
                if s > final_interactions[(tf, tgt)]:
                    final_interactions[(tf, tgt)] = s

    with open(output_file, 'w') as f:
        for (tf, tgt), sig in final_interactions.items():
            f.write(f"{tf}\t{tgt}\t{sig}\n")

# =============================================================================
# MODULE 3: MOTIF SCANNING & FILTERING
# =============================================================================

def run_motif_filtering(target_config, atac_config, motif_meme, motif_list, target_tf_list,
                        pseudo_chip, genie3_file, out_prefix, threads, pvalue, tmp_dir):
    logging.info("--- Starting Motif Scan and Filter ---")

    target_info = {}
    with open(target_config) as f:
        parts = f.readline().strip().split('\t')
        target_info = {'sp': parts[0], 'fasta': parts[1], 'gff': parts[2]}

    atac_map = load_config_map(atac_config)
    if target_info['sp'] not in atac_map:
        logging.error("Target species not found in ATAC config.")
        return

    # 1. Get Promoters and Intersect with ATAC
    tss_list = parse_gff_tss(target_info['gff'])
    promoters = create_promoter_bed(tss_list)
    prom_path = os.path.join(tmp_dir, "target_promoters.bed")
    with open(prom_path, 'w') as f: f.write("\n".join(promoters))

    gene_atac_bed = os.path.join(tmp_dir, "target_gene_atac.bed")

    with open(gene_atac_bed, 'w') as f_out:
        run_command(["bedtools", "intersect", "-a", prom_path, "-b", atac_map[target_info['sp']], "-wa", "-wb"], capture_output_file=f_out)

    regions_bed = os.path.join(tmp_dir, "scan_regions.bed")
    with open(gene_atac_bed) as inf, open(regions_bed, 'w') as outf:
        seen = set()
        for line in inf:
            c = line.strip().split('\t')
            if len(c) > 8:
                entry = f"{c[6]}\t{c[7]}\t{c[8]}\t{c[3]}"
                if entry not in seen:
                    outf.write(entry + "\n")
                    seen.add(entry)

    # 2. Get Fasta
    regions_fasta = os.path.join(tmp_dir, "scan_regions.fa")
    run_command(["bedtools", "getfasta", "-fi", target_info['fasta'], "-bed", regions_bed, "-fo", regions_fasta])

    # 3. FIMO Scan
    fimo_out = os.path.join(tmp_dir, "fimo_out")
    logging.info("Running FIMO...")
    # Error checking for FIMO execution
    if not os.path.exists(motif_meme):
        logging.error(f"Motif file does not exist: {motif_meme}")
        sys.exit(1)

    run_command(["fimo", "--oc", fimo_out, "--parse-genomic-coord", "--thresh", str(pvalue), motif_meme, regions_fasta])

    fimo_file = os.path.join(fimo_out, "fimo.tsv")
    if not os.path.exists(fimo_file):
        logging.warning("FIMO generated no output.")
        return

    # 4. Map Motifs
    motif_fam_map = load_config_map(motif_list)
    gene_allowed_families = defaultdict(set)

    fimo_gff = os.path.join(fimo_out, "fimo.gff")
    intersect_out = os.path.join(tmp_dir, "fimo_gene_intersect.txt")

    with open(intersect_out, 'w') as f_out:
        run_command(["bedtools", "intersect", "-a", fimo_gff, "-b", regions_bed, "-wa", "-wb"], capture_output_file=f_out)

    with open(intersect_out) as f:
        for line in f:
            c = line.strip().split('\t')
            if len(c) < 13: continue
            gene_id = c[12]

            attrs = c[8]
            m_id = ""
            if "Name=" in attrs:
                m_id = attrs.split("Name=")[1].split(";")[0]
            elif "ID=" in attrs:
                m_id = attrs.split("ID=")[1].split(";")[0]

            m_id_clean = m_id.split('_')[0]

            if m_id_clean in motif_fam_map:
                fam = motif_fam_map[m_id_clean]
                gene_allowed_families[gene_id].add(fam)

    # 5. Filter Inputs
    tf_fam_map = load_config_map(target_tf_list)

    def filter_file(input_f, output_f):
        if not input_f or not os.path.exists(input_f): return
        with open(input_f) as inf, open(output_f, 'w') as outf:
            for line in inf:
                c = line.strip().split('\t')
                tf, tgt = c[0], c[1]

                if tf in tf_fam_map:
                    fam = tf_fam_map[tf]
                    if fam in gene_allowed_families.get(tgt, set()):
                        outf.write(line)

    logging.info("Filtering Pseudo-ChIP data...")
    pseudo_chip_filtered_tmp = pseudo_chip + ".filtered"
    pseudo_chip_filtered_out = prefixed_output(out_prefix, "_pseudoChIP_targets_motif_filtered.tsv")
    filter_file(pseudo_chip, pseudo_chip_filtered_tmp)
    shutil.move(pseudo_chip_filtered_tmp, pseudo_chip_filtered_out)

    if genie3_file and os.path.exists(genie3_file):
        logging.info("Filtering GENIE3 data...")
        genie3_filtered_tmp = genie3_file + ".filtered"
        genie3_filtered_out = prefixed_output(out_prefix, "_GENIE3_motif_filtered.tsv")
        filter_file(genie3_file, genie3_filtered_tmp)
        shutil.move(genie3_filtered_tmp, genie3_filtered_out)

# =============================================================================
# MODULE 4: FINAL INTEGRATION
# =============================================================================

def load_matrix(file_path):
    if not os.path.exists(file_path): return pd.DataFrame()
    df = pd.read_csv(file_path, sep='\t', names=['TF', 'Target', 'Weight'])
    return df.pivot(index='Target', columns='TF', values='Weight').fillna(0)

def integrate_features(args, genie3_file, chip_file, motif_filtered_genie3, motif_filtered_chip):
    logging.info("--- Starting Feature Integration ---")

    dfs = {'genie3_raw': None, 'genie3_filt': None, 'chip_raw': None, 'chip_filt': None}

    for weight_str in args.coexpr_chip_motif:
        try:
            parts = weight_str.split(',')
            w_coexpr, w_chip = float(parts[0]), float(parts[1])
            use_motif = parts[2].strip()
        except:
            logging.error(f"Invalid weight format: {weight_str}"); continue

        logging.info(f"Processing: Coexpr={w_coexpr}, ChIP={w_chip}, Motif={use_motif}")

        df_c, df_ch = pd.DataFrame(), pd.DataFrame()

        if w_coexpr > 0:
            key = 'genie3_filt' if use_motif == 'Y' else 'genie3_raw'
            path = motif_filtered_genie3 if use_motif == 'Y' else genie3_file
            if dfs[key] is None: dfs[key] = load_matrix(path)
            df_c = dfs[key]

        if w_chip > 0:
            key = 'chip_filt' if use_motif == 'Y' else 'chip_raw'
            path = motif_filtered_chip if use_motif == 'Y' else chip_file
            if dfs[key] is None: dfs[key] = load_matrix(path)
            df_ch = dfs[key]

        if df_c.empty and df_ch.empty: continue

        idx = df_c.index.union(df_ch.index)
        cols = df_c.columns.union(df_ch.columns)

        mat_c = df_c.reindex(index=idx, columns=cols).fillna(0)
        mat_ch = df_ch.reindex(index=idx, columns=cols).fillna(0)

        final = (mat_c * w_coexpr) + (mat_ch * w_chip)

        base_name = prefixed_output(args.out_prefix, f"_Combine_C{w_coexpr}_Ch{w_chip}_M{use_motif}")
        final.to_hdf(f"{base_name}.h5", key='regulons', mode='w')

        if w_coexpr > 0 and w_chip > 0:
            mask = (mat_c > 0) & (mat_ch > 0)
            overlap = final.where(mask, 0)
            overlap.to_hdf(f"{base_name}_overlap.h5", key='regulons', mode='w')

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PURE Data Processing Pipeline")

    parser.add_argument('--out_prefix', required=True,
                        help="Output file prefix. When --out_dir is provided, only the basename is used inside that directory.")
    parser.add_argument('--out_dir', default=None,
                        help="Directory for all pipeline-generated outputs. Created automatically if it does not exist. Default: current behavior based on --out_prefix.")
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--tmp_dir', help="Custom temp dir")
    parser.add_argument('--dry_run', action='store_true',
                        help="Run non-destructive preflight checks only; do not execute GENIE3, sequence alignment, bedtools, FIMO, or write final outputs.")

    parser.add_argument('--target_genome_config', required=True)
    parser.add_argument('--target_tf_list', required=True)
    parser.add_argument('--chip_ref_genome_config', required=True)
    parser.add_argument('--chip_peak_config', required=True)
    parser.add_argument('--atac_peak_config', required=True)
    parser.add_argument('--motif_list', required=True)
    parser.add_argument('--motif_file', required=True)

    parser.add_argument('--coexpr_chip_motif', nargs='+', required=True, help="List of 'w_coexpr,w_chip,Y/N'")
    parser.add_argument('--chip_species', default=None,
                        help="Comma-separated ChIP reference species to use, e.g. 'A_thaliana,Z_mays,O_sativa'. Default: all species in --chip_ref_genome_config.")
    parser.add_argument('--homology_vote_threshold', type=float, default=0.5,
                        help="Minimum fraction of ChIP-supported TF homologs required to transfer regulation. Default: 0.5, matching the historical behavior.")
    parser.add_argument('--genie3_file', help="Precomputed GENIE3")
    parser.add_argument('--rna_matrix', help="RNA Matrix for GENIE3 calculation")
    parser.add_argument('--aligner', choices=['blast', 'diamond'], default='blast',
                        help="Protein similarity search backend used when no precomputed alignment file is supplied. Choices: blast, diamond. Default: blast.")
    parser.add_argument('--blast_result_file', '--alignment_result_file', dest='blast_result_file', metavar='ALIGNMENT_RESULT_FILE',
                        help="Precomputed tabular protein alignment file with columns: qseqid, sseqid, pident. The --blast_result_file name is retained for backward compatibility; --alignment_result_file is the preferred generic alias.")
    parser.add_argument('--blast_evalue', '--blast_evalue_cutoff', '--alignment_evalue', dest='blast_evalue', metavar='ALIGNMENT_EVALUE', default='1e-20',
                        help="E-value cutoff for generated BLASTP or DIAMOND searches. --alignment_evalue is the preferred generic alias; --blast_evalue and --blast_evalue_cutoff are retained for backward compatibility.")
    parser.add_argument('--genie3_filter', default='q20',
                        help="GENIE3 filtering method passed to the embedded R script, e.g. q10, q20, 50k. Default: q20.")
    parser.add_argument('--motif_scan_pvalue', default='1e-4')

    args = parser.parse_args()

    if not (0.0 <= args.homology_vote_threshold <= 1.0):
        parser.error("--homology_vote_threshold must be between 0 and 1.")
    if args.threads < 1:
        parser.error("--threads must be >= 1.")

    selected_chip_species = parse_species_filter(args.chip_species)

    try:
        configure_output_paths(args, create_dir=True)
    except Exception as exc:
        parser.error(str(exc))

    # 1. Setup Logging (Single file logic)
    setup_logging(prefixed_output(args.out_prefix, ".log"))
    logging.info("Output directory: %s", args.out_dir)
    logging.info("Output prefix: %s", args.out_prefix_name)
    logging.info("Selected sequence aligner: %s", args.aligner)

    if args.dry_run:
        ok = dry_run_preflight(args, selected_chip_species)
        sys.exit(0 if ok else 2)

    # 2. Pre-flight Check (Input Validation)
    validate_input_files(args)

    tmp_base = args.tmp_dir if args.tmp_dir else tempfile.mkdtemp(prefix="PURE_run_")
    if not os.path.exists(tmp_base): os.makedirs(tmp_base)
    logging.info(f"Temporary Directory: {tmp_base}")

    try:
        # Step 1: GENIE3
        genie3_out = args.genie3_file
        if not genie3_out:
            if not args.rna_matrix:
                logging.error("Either --genie3_file or --rna_matrix is required.")
                sys.exit(1)
            genie3_out = prefixed_output(args.out_prefix, "_GENIE3_calculated.tsv")

            r_script_path = os.path.join(tmp_base, "run_genie3.R")
            with open(r_script_path, 'w') as f: f.write(GENIE3_R_SCRIPT_CONTENT)

            cmd = ["Rscript", r_script_path, "--rna_matrix", args.rna_matrix,
                   "--tf_list", args.target_tf_list, "--output", genie3_out,
                   "--threads", str(args.threads), "--filter_method", args.genie3_filter]
            run_command(cmd)

        # Step 2: ChIP Ref Processing
        chip_ref_targets = prefixed_output(args.out_prefix, "_ChIP_ref_targets.tsv")
        process_chip_data(args.chip_ref_genome_config, args.chip_peak_config,
                          args.atac_peak_config, chip_ref_targets, tmp_base,
                          selected_chip_species=selected_chip_species)

        # Step 3: Protein similarity search and orthology transfer
        alignment_file = args.blast_result_file
        if not alignment_file:
            alignment_suffix = "_blast_generated.tsv" if args.aligner == "blast" else "_diamond_generated.tsv"
            alignment_file = prefixed_output(args.out_prefix, alignment_suffix)
            run_sequence_alignment_pipeline(
                args.target_genome_config, args.chip_ref_genome_config,
                alignment_file, args.threads, args.blast_evalue, tmp_base,
                aligner=args.aligner
            )
        else:
            logging.info("Using precomputed alignment file: %s", alignment_file)

        pseudo_chip_raw = prefixed_output(args.out_prefix, "_pseudoChIP_targets_raw.tsv")
        align_orthogroups(chip_ref_targets, alignment_file, args.chip_peak_config,
                          args.target_tf_list, pseudo_chip_raw, args.threads, tmp_base,
                          selected_chip_species=selected_chip_species,
                          vote_threshold=args.homology_vote_threshold)

        # Step 4: Motif Filtering
        run_motif_filtering(args.target_genome_config, args.atac_peak_config, args.motif_file,
                            args.motif_list, args.target_tf_list, pseudo_chip_raw, genie3_out,
                            args.out_prefix, args.threads, args.motif_scan_pvalue, tmp_base)

        filt_genie = prefixed_output(args.out_prefix, "_GENIE3_motif_filtered.tsv")
        filt_chip = prefixed_output(args.out_prefix, "_pseudoChIP_targets_motif_filtered.tsv")

        # Step 5: Integration
        integrate_features(args, genie3_out, pseudo_chip_raw, filt_genie, filt_chip)

        logging.info("PURE Pipeline completed successfully.")

    finally:
        if not args.tmp_dir and os.path.exists(tmp_base):
            shutil.rmtree(tmp_base, ignore_errors=True)
            logging.info("Cleaned up temporary directory.")

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
