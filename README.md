# PURE: Plant Universal Expression and Regulation Explorer

**PURE** is an interpretable machine-learning framework for building plant gene regulatory feature matrices and prioritizing transcription factors (TFs) that explain differential gene expression in a biological contrast.

PURE integrates condition-specific co-expression, orthology-projected TF-binding evidence, motif support, chromatin accessibility, and CatBoost/SHAP interpretation. The framework is designed for both model and non-model plant species, where complete native cistrome resources are often unavailable.

<p align="center">
  <img src="./Figures/PURE_diagram.png" alt="PURE workflow diagram" width="720">
</p>

## What PURE Does

PURE treats genes as observations and candidate TFs as regulatory features. For each target species and biological contrast, it:

1. Builds co-expression evidence from an RNA-seq matrix with GENIE3, or uses a precomputed GENIE3 network.
2. Projects TF-binding evidence from reference ChIP-seq or DAP-seq resources to the target species through protein homology.
3. Applies a configurable homolog voting strategy to reduce unsupported one-to-one transfers.
4. Filters regulatory links with motif evidence in accessible promoter regions when motif resources are provided.
5. Combines co-expression and projected binding evidence into HDF5 regulatory matrices.
6. Trains CatBoost models to classify DEGs and uses SHAP values to rank TFs by their contribution to DEG or pathway-gene prediction.

DEG classification is an internal model-performance check. TF prioritization is performed downstream by summing evidence-supported SHAP contributions across relevant DEG or pathway targets.

## Installation

Clone the repository and create the recommended environment:

```bash
git clone https://github.com/tulab-sjtu/PURE
cd PURE

conda env create -f Script/PURE_env.yml
conda activate PURE_env
```

The data-processing module also calls external tools. Make sure the following commands are available on `PATH` when needed:

- `Rscript` with R packages `argparse`, `GENIE3`, and `data.table`
- `bedtools`
- `fimo` from MEME Suite
- `makeblastdb` and `blastp` for the default BLAST backend, or `diamond` when `--aligner diamond` is used

## Repository Layout

```text
Script/
  PURE_Data_Process.py                 # Build regulatory feature matrices
  PURE_CatBoost_SHAP.py                # DEG prediction and SHAP attribution
  PURE_Contribution_Visualization.py   # Pathway-level TF visualization
  PURE_env.yml                         # Conda environment

Example_data/
  1_process_example_data/              # Data-processing examples
  2_catboost_example_data/             # CatBoost/SHAP examples
  3_visualization_example_data/        # Visualization examples

Figures/
  PURE_diagram.png
  PURE_Web.png
```

## Quick Start

Run a non-destructive preflight check before the full data-processing workflow:

```bash
python Script/PURE_Data_Process.py \
  --out_prefix O_sativa_TF_regulatory_raw \
  --out_dir Results/O_sativa_process \
  --threads 48 \
  --dry_run \
  --target_genome_config Example_data/1_process_example_data/0_O_sativa_genome.config \
  --target_tf_list Example_data/1_process_example_data/0_Os_TF_list_itak.txt \
  --coexpr_chip_motif "0.5,0.5,Y" "0.5,0.5,N" "0,1,Y" "1,0,Y" \
  --rna_matrix Example_data/1_process_example_data/1_Os_RNA-seq_TPM_matrix_10sample.tsv \
  --genie3_filter q10 \
  --atac_peak_config Example_data/1_process_example_data/2_Os_At_Zm_ATAC.config \
  --chip_ref_genome_config Example_data/1_process_example_data/2_At_Zm_genome.config \
  --chip_peak_config Example_data/1_process_example_data/2_At_Zm_peaks_path_q005.config \
  --chip_species A_thaliana,Z_mays \
  --homology_vote_threshold 0.5 \
  --aligner blast \
  --alignment_evalue 1e-20 \
  --motif_list Example_data/1_process_example_data/3_Motif_family.config \
  --motif_file Example_data/1_process_example_data/3_DAP_ChIP_motifs.meme \
  --motif_scan_pvalue 1e-4
```

If the dry run passes, remove `--dry_run` and run the same command to generate the regulatory matrices.

## Input File Formats

All configuration files are tab-delimited text files. Paths in example config files are placeholders and must be edited to match local FASTA, GFF3, peak, motif, and expression files.

### Target Genome Config

`--target_genome_config` describes the target species:

```text
SpeciesID    GenomeFASTA    AnnotationGFF3    ProteinFASTA
```

Example:

```text
O_sativa    /path/to/O_sativa.fa    /path/to/O_sativa.gff3    /path/to/O_sativa.pep.fa
```

### Target TF List

`--target_tf_list` lists candidate TFs in the target species:

```text
TFGeneID    TFFamily
```

The first TF gene ID is also used to infer the target gene-ID prefix for orthology projection.

### RNA Matrix

`--rna_matrix` is a gene-by-sample expression matrix. The first column must contain gene IDs, and the remaining columns should contain numeric expression values such as TPM or normalized counts:

```text
GeneID    Sample1    Sample2    Sample3
GeneA     12.4       9.1        15.0
GeneB     0.0        1.2        0.8
```

GENIE3 is run on `log2(expression + 1)`. If you already have a GENIE3 network, provide it with `--genie3_file` instead of `--rna_matrix`.

### ChIP Reference Genome Config

`--chip_ref_genome_config` uses the same four-column format as the target genome config, but for reference species that provide ChIP-seq or DAP-seq resources:

```text
SpeciesID    GenomeFASTA    AnnotationGFF3    ProteinFASTA
```

Use `--chip_species` to restrict the reference species used in a run. By default, all species in this config are used.

### ChIP or DAP Peak Config

`--chip_peak_config` lists TF-binding peak files:

```text
ReferenceSpeciesID    ReferenceTFGeneID    TFName    TFFamily    PeakBED
```

Peak files should be BED-like files containing genomic intervals and a signal column. PURE associates peaks with accessible promoter regions before projection.

### ATAC Peak Config

`--atac_peak_config` lists accessible chromatin peak files:

```text
SpeciesID    ATACPeakBED
```

Entries are required for the target species and for any selected reference species whose TF-binding peaks will be processed.

### Motif Files

`--motif_file` is a MEME-format motif file used by FIMO. `--motif_list` maps motif IDs to TF families:

```text
MotifID    TFFamily
```

Motif support is used as an optional filtering layer. TFs without motif support are not necessarily removed unless the selected integration setting requests motif-filtered evidence.

## Module 1: Data Processing

The data-processing module creates regulatory feature matrices by combining co-expression, projected binding, and motif evidence.

### Run From Raw RNA-seq Data

```bash
python Script/PURE_Data_Process.py \
  --out_prefix O_sativa_TF_regulatory_raw \
  --out_dir Results/O_sativa_process \
  --threads 48 \
  --target_genome_config Example_data/1_process_example_data/0_O_sativa_genome.config \
  --target_tf_list Example_data/1_process_example_data/0_Os_TF_list_itak.txt \
  --coexpr_chip_motif "0.5,0.5,Y" "0.5,0.5,N" "0,1,Y" "1,0,Y" "0,1,N" "1,0,N" \
  --rna_matrix Example_data/1_process_example_data/1_Os_RNA-seq_TPM_matrix_10sample.tsv \
  --genie3_filter q10 \
  --atac_peak_config Example_data/1_process_example_data/2_Os_At_Zm_ATAC.config \
  --chip_ref_genome_config Example_data/1_process_example_data/2_At_Zm_genome.config \
  --chip_peak_config Example_data/1_process_example_data/2_At_Zm_peaks_path_q005.config \
  --chip_species A_thaliana,Z_mays \
  --homology_vote_threshold 0.5 \
  --aligner blast \
  --alignment_evalue 1e-20 \
  --motif_list Example_data/1_process_example_data/3_Motif_family.config \
  --motif_file Example_data/1_process_example_data/3_DAP_ChIP_motifs.meme \
  --motif_scan_pvalue 1e-4 \
  > Osativa_TF_regulatory_raw.log 2>&1
```

### Run From Intermediate Files

Use this mode when GENIE3 and protein-alignment results have already been generated and you only need to retune evidence weights, reference species, motif filtering, or homolog voting.

```bash
python Script/PURE_Data_Process.py \
  --out_prefix O_sativa_TF_regulatory_mid \
  --out_dir Results/O_sativa_process \
  --threads 48 \
  --target_genome_config Example_data/1_process_example_data/0_O_sativa_genome.config \
  --target_tf_list Example_data/1_process_example_data/0_Os_TF_list_itak.txt \
  --coexpr_chip_motif "0.5,0.5,Y" "0.5,0.5,N" "0,1,Y" "1,0,Y" "0,1,N" "1,0,N" \
  --genie3_file Example_data/1_process_example_data/1_O_sativa_TF_regulatory_GENIE3_q10_normalization.tsv \
  --atac_peak_config Example_data/1_process_example_data/2_Os_At_Zm_ATAC.config \
  --chip_ref_genome_config Example_data/1_process_example_data/2_At_Zm_genome.config \
  --chip_peak_config Example_data/1_process_example_data/2_At_Zm_peaks_path_q005.config \
  --alignment_result_file Example_data/1_process_example_data/2_O_sativa_TF_regulatory_blast_relationship.tsv \
  --chip_species A_thaliana,Z_mays \
  --homology_vote_threshold 0.5 \
  --motif_list Example_data/1_process_example_data/3_Motif_family.config \
  --motif_file Example_data/1_process_example_data/3_DAP_ChIP_motifs.meme \
  --motif_scan_pvalue 1e-4 \
  > Osativa_TF_regulatory_mid.log 2>&1
```

### Key Data-Processing Parameters

| Parameter | Required | Description |
| --- | --- | --- |
| `--out_prefix` | Yes | Output file prefix. If `--out_dir` is provided, only the basename is used inside that directory. |
| `--out_dir` | No | Central output directory for all generated files. It is created automatically. |
| `--threads` | No | Number of CPU threads. Default: `4`. |
| `--tmp_dir` | No | Custom temporary directory. If omitted, PURE creates and removes a temporary run directory. |
| `--dry_run` | No | Checks input files, configs, output paths, external tools, selected species, motif files, and common runtime hazards without running the pipeline. |
| `--target_genome_config` | Yes | Target species genome, annotation, and protein FASTA config. |
| `--target_tf_list` | Yes | Target TF gene IDs and TF families. |
| `--rna_matrix` | Conditional | Expression matrix used to run GENIE3. Required if `--genie3_file` is not provided. |
| `--genie3_file` | Conditional | Precomputed GENIE3 links. Required if `--rna_matrix` is not provided. |
| `--genie3_filter` | No | GENIE3 filter passed to the embedded R script, for example `q10`, `q20`, or `50k`. Default: `q20`. |
| `--chip_ref_genome_config` | Yes | Reference species genome config for ChIP-seq or DAP-seq resources. |
| `--chip_peak_config` | Yes | Reference TF peak files and TF family metadata. |
| `--chip_species` | No | Comma-separated reference species to use, such as `A_thaliana,Z_mays,O_sativa`. Default: all species in `--chip_ref_genome_config`. |
| `--homology_vote_threshold` | No | Minimum fraction of ChIP-supported homologs required to transfer a TF-target link. Default: `0.5`. |
| `--aligner` | No | Protein similarity backend used when no precomputed alignment is supplied. Choices: `blast` or `diamond`. Default: `blast`. |
| `--alignment_result_file` | No | Precomputed three-column alignment table: `qseqid`, `sseqid`, `pident`. Alias: `--blast_result_file`. |
| `--alignment_evalue` | No | E-value cutoff for generated BLASTP or DIAMOND searches. Aliases: `--blast_evalue`, `--blast_evalue_cutoff`. Default: `1e-20`. |
| `--atac_peak_config` | Yes | ATAC/open chromatin peak config for target and reference species. |
| `--motif_list` | Yes | Motif-to-TF-family mapping. |
| `--motif_file` | Yes | MEME-format motif file used by FIMO. |
| `--motif_scan_pvalue` | No | FIMO motif scan threshold. Default: `1e-4`. |
| `--coexpr_chip_motif` | Yes | One or more evidence-combination settings in `CoexprWeight,ChIPWeight,Y/N` format. |

### Understanding `--coexpr_chip_motif`

Each value contains three fields:

```text
CoexpressionWeight,ProjectedBindingWeight,UseMotifFilter
```

Examples:

- `"0.5,0.5,Y"`: combine co-expression and projected binding equally, then use motif-filtered links.
- `"1,0,N"`: use co-expression only without motif filtering.
- `"0,1,Y"`: use motif-filtered projected binding only.

PURE writes one HDF5 matrix for each requested evidence setting.

### Understanding Homolog Voting

`--homology_vote_threshold` controls how strict Ortho-ChIP projection is. For a target-gene candidate, PURE evaluates the fraction of reference TF homologs with observed binding support. The link is transferred only when:

```text
supported_reference_homologs / reference_homologs_with_binding_data >= homology_vote_threshold
```

The default value `0.5` corresponds to majority voting. Higher values are more stringent and may improve precision, while lower values recover broader candidate links but can introduce more weakly supported projections.

### Main Outputs

The output prefix controls all generated file names. With `--out_prefix O_sativa_TF_regulatory_raw --out_dir Results/O_sativa_process`, typical outputs include:

```text
Results/O_sativa_process/O_sativa_TF_regulatory_raw.log
Results/O_sativa_process/O_sativa_TF_regulatory_raw_GENIE3_calculated.tsv
Results/O_sativa_process/O_sativa_TF_regulatory_raw_ChIP_ref_targets.tsv
Results/O_sativa_process/O_sativa_TF_regulatory_raw_blast_generated.tsv
Results/O_sativa_process/O_sativa_TF_regulatory_raw_pseudoChIP_targets_raw.tsv
Results/O_sativa_process/O_sativa_TF_regulatory_raw_GENIE3_motif_filtered.tsv
Results/O_sativa_process/O_sativa_TF_regulatory_raw_pseudoChIP_targets_motif_filtered.tsv
Results/O_sativa_process/O_sativa_TF_regulatory_raw_Combine_C0.5_Ch0.5_MY.h5
```

HDF5 matrices are written with key `/regulons`.

## Module 2: DEG Prediction and SHAP Interpretation

This module trains CatBoost models on the regulatory matrices and calculates TF-level SHAP contributions.

```bash
python Script/PURE_CatBoost_SHAP.py \
  --out_prefix Os_zt4h_vs_zt20h_Result \
  --threads 48 \
  --h5_key "/regulons" \
  --TF_features Example_data/2_catboost_example_data/O_sativa_TF_regulatory.h5 \
  --n_splits 10 \
  --iterations 1500 \
  --learning_rate 0.03 \
  --depth 6 \
  --l2_leaf_reg 3.0 \
  --auto_class_weights Balanced \
  --DEGs Example_data/2_catboost_example_data/Os_zt4h_vs_zt20h_DEG_2col.csv \
  > Os_zt4h_vs_zt20h_CatBoost.log 2>&1
```

`--DEGs` should be a two-column CSV file:

```text
GeneID,Label
GeneA,1
GeneB,0
```

The label represents the expression class used for the contrast, such as positive DEG versus negative DEG, or DEG versus background, depending on the analysis design.

## Module 3: Contribution Visualization

This module converts SHAP contribution matrices into pathway-level TF summaries and publication-ready plots. It can focus on selected pathways, display expression dynamics of key TFs and target genes, and annotate TFs through homology to model species.

```bash
python Script/PURE_Contribution_Visualization.py \
  --out_prefix Os_LvsD_SHAP_Plots \
  --contribution_matrix Example_data/3_visualization_example_data/Os_LvsD_SHAP_exp_0.1121_pos_is_Light_neg_is_Dark.csv \
  --filter_percent 20 \
  --target_tf_list Example_data/3_visualization_example_data/0_Os_TF_list_itak.txt \
  --pathway_config Example_data/3_visualization_example_data/Os_PS_LHC_genes.config \
  --best_hit_to_model_species Example_data/3_visualization_example_data/Os2At_besthit.blast \
  --heatmap_expression Example_data/3_visualization_example_data/Os_zeitgeber_TPM.tsv \
  --model_species_annotation Example_data/3_visualization_example_data/At_annotation.config \
  > Os_LvsD_SHAP_vis.log 2>&1
```

Important visualization options:

- `--filter_percent`: retains the strongest contribution links, such as the top 20 percent.
- `--pathway_config`: focuses the ranking on user-defined pathway or target-gene sets.
- `--heatmap_expression`: adds expression profiles for candidate TFs and pathway genes.
- `--best_hit_to_model_species`: maps target-species genes to a model species.
- `--model_species_annotation`: adds readable functional annotations for homologous model-species genes.

## Extending PURE to Other Species

The web server currently supports representative species prepared by the authors. The command-line workflow is more flexible: users can provide their own target genome, TF list, RNA-seq matrix, ATAC peaks, reference cistrome resources, motifs, and protein alignments.

For a new species, prepare:

1. A target genome config with genome FASTA, GFF3, and protein FASTA.
2. A target TF list with gene IDs and TF families.
3. A gene expression matrix or precomputed GENIE3 network.
4. ATAC or accessible-region BED files for motif filtering.
5. Reference ChIP-seq or DAP-seq peak resources, or a curated subset selected with `--chip_species`.
6. Motif resources in MEME format and a motif-family mapping file.

If the target species is evolutionarily distant from the reference cistrome species, use `--dry_run`, inspect projection coverage, and consider testing several `--homology_vote_threshold` and `--coexpr_chip_motif` settings.

## Web Server

PURE is also available as a web resource:

https://plantencodedb.sjtu.edu.cn/pure/

The web interface supports species selection, DEG and pathway-gene upload, feature-weight adjustment, motif-filter selection, ranked TF inspection, and downloadable results.

<p align="center">
  <img src="./Figures/PURE_Web.png" alt="PURE web interface" width="720">
</p>
