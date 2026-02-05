#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PURE: Pipeline for Uncovering Regulatory Elements (Integrated Data Processing)
Gene Regulatory Network Construction Pipeline

This comprehensive pipeline integrates evidence from co-expression analysis (GENIE3),
cross-species ChIP-seq homology transfer (via BLAST/OrthoGroups), and motif scanning
to construct high-quality, weighted Gene Regulatory Networks (GRNs).

Author: CS Li
Date: 2026-01-31
Version: 0.0.3
"""

import argparse
import os
import sys
import subprocess
import logging
import tempfile
import shutil
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
    
    # Optional files checks
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
    Logs output to file and console as configured.
    """
    cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
    logging.info(f"CMD: {cmd_str}")
    
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

def process_chip_data(chip_ref_config, chip_peak_config, atac_peak_config, output_file, tmp_dir):
    logging.info("--- Starting ChIP Peak Association ---")
    
    ref_genomes = {} 
    with open(chip_ref_config) as f:
        for line in f:
            p = line.strip().split('\t')
            ref_genomes[p[0]] = {'gff': p[2]}

    atac_files = load_config_map(atac_peak_config)
    species_gene_atac_map = {}
    
    for species, genome_data in ref_genomes.items():
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
        
        try:
            with open(gene_atac_bed, 'w') as out_f:
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

    # Process Peaks
    with open(output_file, 'w') as final_out:
        with open(chip_peak_config, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 5: continue
                species, tf_id, tf_name, family, peak_path = parts
                
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
# MODULE 2: BLAST & ORTHOLOGY ALIGNMENT
# =============================================================================

def run_blast_pipeline(target_config, ref_config, output_file, threads, evalue, tmp_dir):
    logging.info("--- Starting BLASTP Pipeline ---")
    
    pep_files = []
    with open(target_config) as f:
        pep_files.append(f.readline().strip().split('\t')[3])
    with open(ref_config) as f:
        for line in f:
            pep_files.append(line.strip().split('\t')[3])
            
    combined_fasta = os.path.join(tmp_dir, "all_prot.fasta")
    with open(combined_fasta, 'w') as outfile:
        for p in pep_files:
            if os.path.exists(p):
                with open(p) as infile: outfile.write(infile.read())
                    
    db_path = os.path.join(tmp_dir, "blastdb")
    run_command(["makeblastdb", "-in", combined_fasta, "-dbtype", "prot", "-out", db_path])
    
    logging.info("Running BLASTP...")
    run_command([
        "blastp", "-query", combined_fasta, "-db", db_path, "-out", output_file,
        "-evalue", evalue, "-num_threads", str(threads), "-outfmt", "6 qseqid sseqid pident"
    ])
    
    df = pd.read_csv(output_file, sep='\t', names=['q', 's', 'p'])
    df = df[df['q'] != df['s']]
    df.to_csv(output_file, sep='\t', index=False, header=False)

def build_orthogroups(blast_file, relevant_tfs, tmp_dir):
    logging.info("Building TF OrthoGroups...")
    adj = defaultdict(set)
    relevant_set = set(relevant_tfs)
    
    with open(blast_file, 'r') as f:
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

def process_homology_voting_chunk(chunk_targets, groups, blast_file, chip_data_file, target_species_prefix):
    homologs = defaultdict(set)
    with open(blast_file, 'r') as f:
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
                if t_c > 0 and (v_c / t_c) >= 0.5:
                    for member in members:
                        if member.startswith(target_species_prefix):
                            results.append((member, target_gene, local_max_sig))
    return results

def align_orthogroups(chip_target_file, blast_file, chip_peak_config, target_tf_list, output_file, threads, tmp_dir):
    logging.info("--- Starting OrthoGroup Alignment ---")
    
    with open(target_tf_list) as f:
        first_tf = f.readline().split('\t')[0]
        import re
        m = re.match(r'^([^0-9]+)', first_tf)
        if not m: raise ValueError("Cannot determine species prefix from TF list")
        target_prefix = m.group(1)
        
    tfs = set()
    with open(target_tf_list) as f:
        for line in f: tfs.add(line.split('\t')[0])
    with open(chip_peak_config) as f:
        for line in f: tfs.add(line.split('\t')[1])
        
    groups = build_orthogroups(blast_file, list(tfs), tmp_dir)
    logging.info(f"Generated {len(groups)} OrthoGroups.")
    
    target_genes = set()
    with open(blast_file) as f:
        for line in f:
            q, s, _ = line.strip().split('\t')
            if q.startswith(target_prefix): target_genes.add(q)
            if s.startswith(target_prefix): target_genes.add(s)
    target_genes = sorted(list(target_genes))
    
    chunk_size = len(target_genes) // threads + 1
    chunks = [target_genes[i:i + chunk_size] for i in range(0, len(target_genes), chunk_size)]
    
    logging.info(f"Processing in {threads} threads...")
    worker = partial(process_homology_voting_chunk, 
                     groups=groups, blast_file=blast_file, 
                     chip_data_file=chip_target_file, target_species_prefix=target_prefix)
    
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
    filter_file(pseudo_chip, pseudo_chip + ".filtered")
    shutil.move(pseudo_chip + ".filtered", os.path.join(os.path.dirname(pseudo_chip), f"{out_prefix}_pseudoChIP_targets_motif_filtered.tsv"))
    
    if genie3_file and os.path.exists(genie3_file):
        logging.info("Filtering GENIE3 data...")
        filter_file(genie3_file, genie3_file + ".filtered")
        shutil.move(genie3_file + ".filtered", os.path.join(os.path.dirname(genie3_file), f"{out_prefix}_GENIE3_motif_filtered.tsv"))

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
        
        base_name = f"{args.out_prefix}_Combine_C{w_coexpr}_Ch{w_chip}_M{use_motif}"
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
    
    parser.add_argument('--out_prefix', required=True, help="Prefix for output files")
    parser.add_argument('--threads', type=int, default=4, help="Number of threads")
    parser.add_argument('--tmp_dir', help="Custom temp dir")
    
    # Required Configs
    parser.add_argument('--target_genome_config', required=True, help="Target species genome config")
    parser.add_argument('--target_tf_list', required=True, help="Target species TF list")
    parser.add_argument('--chip_ref_genome_config', required=True, help="Reference species genome config")
    parser.add_argument('--chip_peak_config', required=True, help="ChIP-seq peak config")
    parser.add_argument('--atac_peak_config', required=True, help="ATAC-seq peak config")
    parser.add_argument('--motif_list', required=True, help="Motif family mapping")
    parser.add_argument('--motif_file', required=True, help="Motif meme file")
    parser.add_argument('--coexpr_chip_motif', nargs='+', required=True, help="List of 'w_coexpr,w_chip,Y/N'")
    
    # Optional / Conditional
    parser.add_argument('--genie3_file', help="Precomputed GENIE3 file")
    parser.add_argument('--rna_matrix', help="RNA Matrix for GENIE3 calculation")
    parser.add_argument('--genie3_filter', default='q20', help="Filter method for GENIE3 (e.g., q20, 100k)")
    parser.add_argument('--blast_result_file', help="Precomputed BLAST file")
    parser.add_argument('--blast_evalue', default='1e-20', help="BLAST E-value cutoff")
    parser.add_argument('--motif_scan_pvalue', default='1e-4', help="FIMO p-value cutoff")
    
    args = parser.parse_args()
    
    # 1. Setup Logging
    setup_logging(f"{args.out_prefix}.log")
    
    # 2. Pre-flight Check
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
            genie3_out = f"{args.out_prefix}_GENIE3_calculated.tsv"
            
            r_script_path = os.path.join(tmp_base, "run_genie3.R")
            with open(r_script_path, 'w') as f: f.write(GENIE3_R_SCRIPT_CONTENT)
            
            cmd = ["Rscript", r_script_path, "--rna_matrix", args.rna_matrix, 
                   "--tf_list", args.target_tf_list, "--output", genie3_out, 
                   "--threads", str(args.threads), "--filter_method", args.genie3_filter]
            run_command(cmd)
        
        # Step 2: ChIP Ref Processing
        chip_ref_targets = f"{args.out_prefix}_ChIP_ref_targets.tsv"
        process_chip_data(args.chip_ref_genome_config, args.chip_peak_config, 
                          args.atac_peak_config, chip_ref_targets, tmp_base)
        
        # Step 3: BLAST & Orthology
        blast_file = args.blast_result_file
        if not blast_file:
            blast_file = f"{args.out_prefix}_blast_generated.tsv"
            run_blast_pipeline(args.target_genome_config, args.chip_ref_genome_config, 
                               blast_file, args.threads, args.blast_evalue, tmp_base)
            
        pseudo_chip_raw = f"{args.out_prefix}_pseudoChIP_targets_raw.tsv"
        align_orthogroups(chip_ref_targets, blast_file, args.chip_peak_config, 
                          args.target_tf_list, pseudo_chip_raw, args.threads, tmp_base)
        
        # Step 4: Motif Filtering
        run_motif_filtering(args.target_genome_config, args.atac_peak_config, args.motif_file,
                            args.motif_list, args.target_tf_list, pseudo_chip_raw, genie3_out,
                            args.out_prefix, args.threads, args.motif_scan_pvalue, tmp_base)
        
        filt_genie = f"{args.out_prefix}_GENIE3_motif_filtered.tsv"
        filt_chip = f"{args.out_prefix}_pseudoChIP_targets_motif_filtered.tsv"
        
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