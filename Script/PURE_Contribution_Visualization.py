#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PURE: Post-analysis Utility for Regulatory network Explanations
Contribution and Expression Visualization Script.

This comprehensive visualization toolkit processes SHAP contribution matrices and 
gene expression data to generate high-quality, publication-ready figures for 
Gene Regulatory Network (GRN) analysis.

Key Features:
    1. Contribution Dot Plots: Visualizes summed TF contributions (Overall & Pathway-specific).
    2. Regulatory Count Analysis: Stacked bar plots of regulation directionality.
    3. Waterfall Plots: Precise, horizontal visualization of top contributors per gene.
    4. Expression Heatmaps: Clustered analysis of gene and TF expression patterns.
    5. Contribution Heatmaps: Z-score normalized regulatory strength visualization.
    6. Network Topology Visualization: concentric layout for pathway-specific TF-Gene interactions.

Author: CS Li
Date: 2026-01-31
Version: 0.0.1
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.path as mpath
import seaborn as sns
import networkx as nx
import re
import os
import sys
from datetime import datetime

# --- Configuration & Styling --- #

def log_message(message):
    """Prints a standardized log message with a timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: {message}")

def set_plot_style():
    """Configures global matplotlib settings for publication-quality vector graphics."""
    plt.style.use('seaborn-v0_8-white')
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['savefig.format'] = 'pdf'
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'

# --- Data Processing Modules --- #

def filter_contribution_by_percentile(df, top_percent):
    """Filters the contribution matrix to retain only the most significant interactions."""
    if not 0 < top_percent <= 100:
        log_message(f"Filter percentile {top_percent} is out of range (0-100]. Skipping filter.")
        return df

    log_message(f"Filtering matrix to retain top {top_percent}% of absolute contribution signals.")
    
    # Extract non-zero values to determine threshold
    non_zero_values = df.abs().values[np.nonzero(df.values)]
    if non_zero_values.size == 0:
        log_message("Matrix contains no non-zero values. Filtering skipped.")
        return df

    threshold = np.percentile(non_zero_values, 100 - top_percent)
    log_message(f"Contribution threshold set at: {threshold:.4g}")

    # Apply mask
    df_filtered = df.where(df.abs() >= threshold, 0)
    
    kept_count = (df_filtered != 0).sum().sum()
    total_count = len(non_zero_values)
    log_message(f"Filter complete. Retained {kept_count}/{total_count} interactions.")
    
    return df_filtered

def annotate_tf_names(df, blast_file, annotation_file, tf_family_file):
    """Annotates TF identifiers (columns) with readable gene names and family classifications."""
    log_message("Initializing TF annotation pipeline...")
    
    # Load Family Data
    family_map = {}
    if tf_family_file:
        try:
            # Using raw string for regex separator to ensure python version compatibility
            fam_df = pd.read_csv(tf_family_file, sep=r'\s+', header=None, usecols=[0, 1], names=['tf_id', 'family'])
            family_map = fam_df.set_index('tf_id')['family'].to_dict()
        except Exception as e:
            log_message(f"TF Family annotation skipped. Reason: {e}")

    # Load Gene Name Data
    gene_name_map = {}
    if blast_file and annotation_file:
        try:
            blast_map_df = pd.read_csv(blast_file, sep='\t', header=None, usecols=[0, 1], names=['query_id', 'subject_id'])
            # Resolve many-to-one mapping by keeping first hit
            blast_map = blast_map_df.drop_duplicates(subset=['query_id'], keep='first').set_index('query_id')['subject_id']
            
            annot_map_df = pd.read_csv(annotation_file, sep='\t', usecols=[0, 1], names=['geneID', 'GeneName'], skiprows=1)
            annot_map = annot_map_df.drop_duplicates(subset=['geneID'], keep='first').set_index('geneID')['GeneName']
            
            gene_name_map = blast_map.map(annot_map).dropna().to_dict()
        except Exception as e:
            log_message(f"Gene name mapping skipped. Reason: {e}")

    # Construct New Column Names
    new_column_map = {}
    for tf_id in df.columns:
        components = [tf_id]
        if tf_id in gene_name_map:
            components.append(gene_name_map[tf_id])
        if tf_id in family_map:
            components.append(family_map[tf_id])
        new_column_map[tf_id] = "_".join(components)

    log_message("Annotation pipeline completed.")
    return df.rename(columns=new_column_map), new_column_map

def parse_metadata_from_filename(filename):
    """Extracts experimental parameters (Baseline value, Direction labels) from the filename."""
    # Extract Expected Value (Baseline)
    base_val = 0.0
    exp_match = re.search(r'_exp_([-_0-9.]+)_', filename)
    if exp_match:
        try:
            base_val = float(exp_match.group(1))
        except ValueError:
            pass

    # Extract Direction Labels
    pos_label = "Upregulated"
    neg_label = "Downregulated"
    pos_match = re.search(r'pos_is_([a-zA-Z0-9]+)', filename)
    neg_match = re.search(r'neg_is_([a-zA-Z0-9]+)', filename)
    
    if pos_match: pos_label = pos_match.group(1)
    if neg_match: neg_label = neg_match.group(1)
    
    return base_val, pos_label, neg_label

# --- Visualization Engines --- #

def plot_summed_contribution_dotplot(df, top_n, output_dir, pos_label, neg_label, pathway_name=None):
    """Generates a dot plot of TFs sorted by their total summed contribution."""
    if df.empty: return []
    
    context = pathway_name if pathway_name else "Overall"
    log_message(f"Plotting summed contributions for: {context} (Top {top_n})")

    tf_sums = df.sum().sort_values(ascending=False)
    
    # Select Top N Positive and Top N Negative contributors
    top_pos = tf_sums[tf_sums > 0].head(top_n)
    top_neg = tf_sums[tf_sums < 0].tail(top_n)
    top_tfs = pd.concat([top_pos, top_neg]).sort_values(ascending=False)
    
    if top_tfs.empty: return []

    # Prepare Plot Data
    y_pos = np.arange(len(top_tfs))
    values = top_tfs.values
    labels = top_tfs.index
    
    # Dynamic Sizing
    fig_height = max(6, len(top_tfs) * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    # Normalization & Colormap
    norm = mcolors.TwoSlopeNorm(vmin=values.min(), vcenter=0, vmax=values.max())
    cmap = sns.color_palette("vlag", as_cmap=True)
    
    # Size Scaling
    abs_vals = np.abs(values)
    size_min, size_max = 50, 450
    sizes = size_min + (abs_vals - abs_vals.min()) / (abs_vals.max() - abs_vals.min() + 1e-9) * (size_max - size_min)

    sc = ax.scatter(values, y_pos, s=sizes, c=values, cmap=cmap, norm=norm, 
                    alpha=0.8, edgecolors="black", linewidth=0.5)

    # Decoration
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(f"Sum of SHAP Contributions\n(Positive → {pos_label}, Negative → {neg_label})", 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel("Transcription Factors", fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.axvline(0, color='grey', linestyle='--', linewidth=1)
    
    title = f"Top {top_n} TF Contributions"
    if pathway_name: title += f" ({pathway_name})"
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.03).set_label('Contribution Value', fontsize=10)

    # Save
    clean_name = context.replace(" ", "_").replace("/", "-")
    filename = f"SummedContribution_DotPlot_{clean_name}_Top{top_n}.pdf"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
    
    return top_tfs.index.tolist()

def plot_contribution_count_barplot(df, top_n, output_dir, pos_label, neg_label, pathway_name=None):
    """Generates a stacked bar plot showing the frequency of regulation (count of genes targeted)."""
    if df.empty: return
    
    context = pathway_name if pathway_name else "Overall"
    log_message(f"Plotting regulation counts for: {context}")

    pos_counts = (df > 0).sum()
    neg_counts = (df < 0).sum()
    
    # Select TFs with highest total interactions
    top_tfs = (pos_counts + neg_counts).nlargest(top_n)
    if top_tfs.empty: return

    plot_df = pd.DataFrame({
        f'Positive ({pos_label})': pos_counts[top_tfs.index],
        f'Negative ({neg_label})': neg_counts[top_tfs.index]
    })
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = [sns.color_palette("vlag", as_cmap=True)(0.9), sns.color_palette("vlag", as_cmap=True)(0.1)]
    
    plot_df.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.8)
    
    ax.set_ylabel("Number of Regulated Genes", fontsize=12, fontweight='bold')
    ax.set_xlabel("Transcription Factors", fontsize=12, fontweight='bold')
    ax.set_title(f"Top {top_n} TFs by Target Gene Count ({context})", fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', rotation=90, labelsize=10)
    
    clean_name = context.replace(" ", "_").replace("/", "-")
    filename = f"ContributionCount_BarPlot_{clean_name}_Top{top_n}.pdf"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

def create_arrow_patch(start, end, y_center, fixed_arrow_width, height=0.7):
    """Helper to create custom arrow shapes for waterfall plots."""
    y1, y2 = y_center - height / 2, y_center + height / 2
    
    if end > start: 
        body_end = max(end - fixed_arrow_width, start)
    else: 
        body_end = min(end + fixed_arrow_width, start)
        
    verts = [
        (start, y1), (body_end, y1), (end, y_center), 
        (body_end, y2), (start, y2), (start, y1)
    ]
    codes = [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.LINETO, 
             mpath.Path.LINETO, mpath.Path.LINETO, mpath.Path.CLOSEPOLY]
    
    return patches.PathPatch(mpath.Path(verts, codes), edgecolor='none')

def plot_waterfall(series, base_value, gene_name, output_dir, pos_label, neg_label, pathway_name=None):
    """Creates a detailed horizontal waterfall plot for a specific gene's regulation."""
    contributions = series.loc[series.abs().nlargest(10).index]
    if contributions.empty: return

    bar_data = {}
    cumulative = base_value
    
    # Calculate segments
    for tf, val in contributions.iloc[::-1].items():
        bar_data[tf] = {'start': cumulative, 'end': cumulative + val, 'val': val}
        cumulative += val
    
    # Determine bounds
    all_x = [base_value, cumulative] + [v for item in bar_data.values() for v in (item['start'], item['end'])]
    x_min, x_max = min(all_x), max(all_x)
    x_range = x_max - x_min
    fixed_arrow_width = x_range * 0.025 if x_range > 0 else 0.01

    fig, ax = plt.subplots(figsize=(8, 5.5))
    palette = sns.color_palette("vlag", 2)
    
    for i, tf in enumerate(contributions.index):
        d = bar_data[tf]
        # Reverse Y index for top-down plotting
        y_idx = len(contributions) - 1 - i
        
        arrow = create_arrow_patch(d['start'], d['end'], y_idx, fixed_arrow_width)
        color = palette[0] if d['val'] > 0 else palette[1]
        arrow.set_facecolor(color)
        ax.add_patch(arrow)

    # Layout adjustment
    padding = x_range * 0.05
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(-0.7, len(contributions) - 0.3)
    ax.axvline(base_value, color='grey', linestyle='--', linewidth=1.5)
    
    ax.set_yticks(range(len(contributions))[::-1])
    ax.set_yticklabels(contributions.index)
    ax.set_xlabel(f"SHAP Value ({pos_label} / {neg_label})", fontweight='bold')
    
    title = f"Waterfall: {gene_name}"
    if pathway_name: title += f" ({pathway_name})"
    ax.set_title(title, fontweight='bold')
    
    # Filename generation
    fname_parts = [gene_name.replace("/", "-"), "Waterfall"]
    if pathway_name: fname_parts.insert(1, pathway_name.replace(" ", "_").replace("/", "-"))
    
    plt.savefig(os.path.join(output_dir, f"{'_'.join(fname_parts)}.pdf"))
    plt.close(fig)

def plot_expression_heatmap(expression_df, id_list, name_map, output_dir, file_prefix, title):
    """Generates a clustered heatmap for gene expression data."""
    valid_ids = [i for i in id_list if i in expression_df.index]
    if not valid_ids: return

    subset = expression_df.loc[valid_ids].rename(index=name_map)
    height = max(6, len(valid_ids) * 0.3)
    width = max(8, subset.shape[1] * 0.5)

    try:
        g = sns.clustermap(subset, method='ward', z_score=0, cmap='coolwarm', 
                           figsize=(width, height), yticklabels=True)
        g.fig.suptitle(title, fontweight='bold', y=1.02)
        plt.savefig(os.path.join(output_dir, f"ExpressionHeatmap_{file_prefix}.pdf"))
        plt.close(g.fig)
    except Exception as e:
        log_message(f"Heatmap generation failed for {file_prefix}. Error: {e}")

def plot_contribution_heatmap(contrib_df, gene_name_map, output_dir, file_prefix, title):
    """Generates a clustered heatmap of SHAP contributions for top TFs."""
    if contrib_df.empty: return

    # Identify Top 40 TFs across the subset
    top_tfs = contrib_df.sum(axis=0).abs().nlargest(40).index
    df = contrib_df[top_tfs]
    
    # Remove rows (genes) with zero signal
    df = df.loc[(df != 0).any(axis=1), :].rename(index=gene_name_map)
    if df.empty: return

    height = max(8, df.shape[0] * 0.3)
    width = max(10, df.shape[1] * 0.4)

    try:
        g = sns.clustermap(df, method='ward', z_score=0, cmap='vlag', 
                           figsize=(width, height), yticklabels=True)
        g.fig.suptitle(f"Top {len(top_tfs)} {title}", fontweight='bold', y=1.02)
        plt.savefig(os.path.join(output_dir, f"ContributionHeatmap_{file_prefix}.pdf"))
        plt.close(g.fig)
    except Exception as e:
        log_message(f"Contribution heatmap failed for {file_prefix}. Error: {e}")

# --- Network Visualization (Concentric/Shell Layout) --- #

def plot_pathway_network(df, gene_map, output_dir, pathway_name, top_n_tfs=20):
    """
    Generates a directed network visualization for a specific pathway.
    
    Layout Algorithm:
        Uses a Concentric Shell Layout to prevent node overlapping.
        - Inner Shell: Transcription Factors (TFs)
        - Outer Shell: Target Genes
    """
    log_message(f"Generating Network Topology for Pathway: {pathway_name}")
    
    # 1. TF Selection: Filter for the most impactful TFs in this pathway
    tf_sums = df.abs().sum().sort_values(ascending=False)
    top_tfs = tf_sums.head(top_n_tfs).index.tolist()
    
    if not top_tfs:
        log_message("No TFs available for network generation. Skipping.")
        return

    # 2. Edge Construction
    df_subset = df[top_tfs]
    edges = []
    
    # Iterate genes (rows) and TFs (cols)
    for target_gene_id, row in df_subset.iterrows():
        target_label = gene_map.get(target_gene_id, target_gene_id)
        
        for source_tf, shap_val in row.items():
            if shap_val != 0:
                edges.append({
                    'source': source_tf, 
                    'target': target_label, 
                    'weight': shap_val,
                    'abs_weight': abs(shap_val)
                })
    
    if not edges:
        log_message("No significant edges found for network plot.")
        return

    # 3. Graph Initialization
    G = nx.DiGraph()
    tf_nodes = set()
    gene_nodes = set()
    
    edge_colors = []
    edge_widths = []
    
    all_weights = [e['abs_weight'] for e in edges]
    min_w, max_w = min(all_weights), max(all_weights)
    
    for e in edges:
        G.add_edge(e['source'], e['target'])
        tf_nodes.add(e['source'])
        gene_nodes.add(e['target'])
        
        # Color: Red (Positive), Blue (Negative)
        color = "#d53e4f" if e['weight'] > 0 else "#3288bd"
        edge_colors.append(color)
        
        # Scale width: 1.0 to 5.0
        if max_w != min_w:
            w = 1.0 + 4.0 * (e['abs_weight'] - min_w) / (max_w - min_w)
        else:
            w = 2.0
        edge_widths.append(w)
        
    # 4. Layout Calculation: Concentric Shells
    # Sorting ensures consistent placement order
    tf_list = sorted(list(tf_nodes))
    gene_list = sorted(list(gene_nodes))
    
    # nlist defines the rings: [Inner Ring (TFs), Outer Ring (Genes)]
    pos = nx.shell_layout(G, nlist=[tf_list, gene_list])
    
    # 5. Rendering
    plt.figure(figsize=(14, 14)) # Large canvas to minimize text collision
    ax = plt.gca()
    
    # Draw Edges with Curvature
    nx.draw_networkx_edges(G, pos, 
                           width=edge_widths, 
                           edge_color=edge_colors, 
                           alpha=0.6, 
                           arrows=True, 
                           arrowstyle='-|>', 
                           arrowsize=15, 
                           connectionstyle="arc3,rad=0.1")
    
    # Draw Nodes: TFs (Triangles)
    nx.draw_networkx_nodes(G, pos, 
                           nodelist=tf_list, 
                           node_shape='^', 
                           node_color='#fdae61', 
                           edgecolors='#d73027', 
                           node_size=600, 
                           linewidths=2.0, 
                           label="TF")
    
    # Draw Nodes: Genes (Circles)
    nx.draw_networkx_nodes(G, pos, 
                           nodelist=gene_list, 
                           node_shape='o', 
                           node_color='#abd9e9', 
                           edgecolors='#4575b4', 
                           node_size=300, 
                           linewidths=2.0, 
                           label="Target Gene")
    
    # Draw Labels with Offset
    label_pos = {k: (v[0], v[1] + 0.05) for k, v in pos.items()}
    nx.draw_networkx_labels(G, label_pos, 
                            font_size=9, 
                            font_family='sans-serif', 
                            font_weight='bold',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', label='TF', 
               markerfacecolor='#fdae61', markeredgecolor='#d73027', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Target Gene', 
               markerfacecolor='#abd9e9', markeredgecolor='#4575b4', markersize=12),
        Line2D([0], [0], color='#d53e4f', lw=2, label='Positive Regulation'),
        Line2D([0], [0], color='#3288bd', lw=2, label='Negative Regulation')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=10)
    
    safe_pathway_name = pathway_name.replace(" ", "_").replace("/", "-")
    title_str = f"Regulatory Network: {pathway_name}\n(Top {top_n_tfs} Key TFs)"
    plt.title(title_str, fontsize=18, fontweight='bold', pad=20)
    plt.axis('off')
    
    filename = f"Network_{safe_pathway_name}.pdf"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    log_message(f"Network topology saved to {output_path}")
    plt.close()

# --- Main Execution Flow --- #

def main():
    parser = argparse.ArgumentParser(
        description="PURE Visualization Toolkit (v0.0.1)",
        formatter_class=argparse.RawTextHelpFormatter)
        
    parser.add_argument("--out_prefix", required=True, 
                        help="Output directory for generated plots.")
    parser.add_argument("--contribution_matrix", required=True, 
                        help="Path to SHAP contribution matrix (CSV).")
    
    # Optional Inputs
    parser.add_argument("--gene_config", 
                        help="TSV file for specific genes of interest (id, name).")
    parser.add_argument("--pathway_config", 
                        help="TSV file for pathway mapping (id, name, Pathway).")
    parser.add_argument("--target_tf_list", 
                        help="File containing TF Family information.")
    parser.add_argument("--best_hit_to_model_species", 
                        help="BLAST results for homology-based annotation.")
    parser.add_argument("--model_species_annotation", 
                        help="Model species gene ID to Name mapping.")
    parser.add_argument("--heatmap_expression", 
                        help="Expression matrix (TSV) for heatmap generation.")
    parser.add_argument("--filter_percent", type=float, 
                        help="Percentile threshold to filter weak interactions.")

    args = parser.parse_args()

    log_message("--- Starting PURE Visualization Pipeline ---")
    set_plot_style()
    
    # Directory Setup
    try:
        os.makedirs(args.out_prefix, exist_ok=True)
    except OSError as e:
        log_message(f"Critical Error: Cannot create output directory. {e}")
        sys.exit(1)

    # Data Loading
    try:
        contrib_df = pd.read_csv(args.contribution_matrix, index_col=0)
    except FileNotFoundError:
        log_message("Critical Error: Contribution matrix file not found.")
        sys.exit(1)

    # Pre-processing
    if args.filter_percent:
        contrib_df = filter_contribution_by_percentile(contrib_df, args.filter_percent)

    expression_df = None
    if args.heatmap_expression:
        try:
            expression_df = pd.read_csv(args.heatmap_expression, sep='\t', index_col=0)
        except Exception:
            log_message("Expression data could not be loaded. Skipping related plots.")

    # Metadata Extraction
    matrix_filename = os.path.basename(args.contribution_matrix)
    base_value, pos_label, neg_label = parse_metadata_from_filename(matrix_filename)
    
    # TF Annotation
    contrib_df, tf_name_map = annotate_tf_names(
        contrib_df, 
        args.best_hit_to_model_species, 
        args.model_species_annotation, 
        args.target_tf_list
    )
    tf_annotated_to_original_id = {v: k for k, v in tf_name_map.items()}

    # --- Analysis Phase 1: Global Statistics ---
    top_10_tfs = plot_summed_contribution_dotplot(contrib_df, 10, args.out_prefix, pos_label, neg_label)
    top_20_tfs = plot_summed_contribution_dotplot(contrib_df, 20, args.out_prefix, pos_label, neg_label)
    plot_contribution_count_barplot(contrib_df, 20, args.out_prefix, pos_label, neg_label)

    if expression_df is not None and top_10_tfs:
        ids = [tf_annotated_to_original_id.get(n) for n in top_10_tfs if n in tf_annotated_to_original_id]
        plot_expression_heatmap(expression_df, ids, tf_name_map, args.out_prefix, 
                                "Overall_Top10_TFs", "Expression of Top 10 Overall TFs")

    # --- Analysis Phase 2: Gene-Specific Plots ---
    if args.gene_config:
        try:
            gene_map_df = pd.read_csv(args.gene_config, sep='\t', header=None, 
                                      names=['geneID', 'geneName'], index_col=0)
            gene_name_map = gene_map_df['geneName'].to_dict()
            genes_in_matrix = [gid for gid in gene_name_map.keys() if gid in contrib_df.index]
            
            if expression_df is not None:
                plot_expression_heatmap(expression_df, genes_in_matrix, gene_name_map, 
                                        args.out_prefix, "Interesting_Genes", "Expression of Interesting Genes")

            for gene_id in genes_in_matrix:
                plot_waterfall(contrib_df.loc[gene_id], base_value, gene_name_map[gene_id], 
                               args.out_prefix, pos_label, neg_label)
        except FileNotFoundError:
            log_message("Gene configuration file not found.")

    # --- Analysis Phase 3: Pathway-Specific Analysis ---
    if args.pathway_config:
        log_message(f"--- Initiating Pathway Analysis: {args.pathway_config} ---")
        try:
            pathway_df = pd.read_csv(args.pathway_config, sep='\t')
            # Create GeneID -> GeneName map for this pathway set
            gene_name_map = pd.Series(pathway_df.geneName.values, 
                                      index=pathway_df.geneID).drop_duplicates().to_dict()
            
            for pathway_name, group_df in pathway_df.groupby('Pathway'):
                log_message(f"Analyzing: {pathway_name}")
                pathway_gene_ids = group_df['geneID'].tolist()
                
                # Subset Matrix
                pathway_contrib_df = contrib_df[contrib_df.index.isin(pathway_gene_ids)]
                if pathway_contrib_df.empty: continue
                
                safe_name = pathway_name.replace(" ", "_").replace("/", "-")
                
                # Generate Plots
                p_top10 = plot_summed_contribution_dotplot(pathway_contrib_df, 10, args.out_prefix, 
                                                           pos_label, neg_label, pathway_name)
                plot_summed_contribution_dotplot(pathway_contrib_df, 20, args.out_prefix, 
                                                 pos_label, neg_label, pathway_name)
                plot_contribution_count_barplot(pathway_contrib_df, 20, args.out_prefix, 
                                                pos_label, neg_label, pathway_name)
                plot_contribution_heatmap(pathway_contrib_df, gene_name_map, args.out_prefix, 
                                          f"Pathway_{safe_name}_Genes", f"Contributions in {pathway_name}")

                # Network Visualization (Concentric Layout)
                plot_pathway_network(pathway_contrib_df, gene_name_map, args.out_prefix, 
                                     pathway_name, top_n_tfs=20)

                # Expression Plots for Pathway
                if expression_df is not None:
                    pathway_genes_in_matrix = pathway_contrib_df.index.tolist()
                    plot_expression_heatmap(expression_df, pathway_genes_in_matrix, gene_name_map, 
                                            args.out_prefix, f"Pathway_{safe_name}_Genes", 
                                            f"Expression in {pathway_name}")
                    
                    if p_top10:
                        ids = [tf_annotated_to_original_id.get(n) for n in p_top10 if n in tf_annotated_to_original_id]
                        plot_expression_heatmap(expression_df, ids, tf_name_map, args.out_prefix, 
                                                f"Pathway_{safe_name}_Top10_TFs", 
                                                f"TF Expression for {pathway_name}")

                # Waterfall Plots for Pathway Genes
                for gene_id in pathway_contrib_df.index:
                    gname = gene_name_map.get(gene_id, gene_id)
                    plot_waterfall(pathway_contrib_df.loc[gene_id], base_value, gname, 
                                   args.out_prefix, pos_label, neg_label, pathway_name)

        except FileNotFoundError:
            log_message("Pathway configuration file not found.")

    log_message("--- Visualization Pipeline Completed Successfully ---")

if __name__ == '__main__':
    main()