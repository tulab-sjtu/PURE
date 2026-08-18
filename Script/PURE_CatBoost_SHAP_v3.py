#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CatBoost/TreeSHAP analysis with repeated-CV and true refit stability.

This is a backward-compatible extension of ``PURE_CatBoost_SHAP_original.py``.
The legacy final model, raw SHAP, zero-input-filtered SHAP, performance CSV, ROC
plots, and output names are retained.  Models use raw features so preprocessing
cannot leak across validation folds.  Stability outputs are additional.  SHAP
values are additive contributions in CatBoost raw-margin space;
a positive/negative value only pushes a prediction toward/away from encoded class
1 and must not be assigned a biological regulatory-direction interpretation.
"""

import argparse
import itertools
import json
import logging

# These dependencies can otherwise emit large volumes of unrelated INFO messages
# after the application logging configuration is installed.
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import os
import platform
import sys
import time
import warnings
from importlib import metadata as importlib_metadata

import catboost as cb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import scipy
from scipy.stats import kendalltau, spearmanr
import seaborn as sns
import shap
import sklearn
from sklearn.metrics import (accuracy_score, auc, balanced_accuracy_score,
                             f1_score, precision_score, recall_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

LOGGER = logging.getLogger("PURE_CatBoost_SHAP_v3")


def log_message(message):
    """Log a timestamped progress message (legacy-compatible visible logging)."""
    LOGGER.info(message)


def set_plot_style():
    """Apply a restrained publication-oriented plotting style."""
    try:
        plt.style.use("seaborn-v0_8-ticks")
    except OSError:
        plt.style.use("default")
    sns.set_context("notebook")
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "xtick.direction": "out", "ytick.direction": "out",
                         "legend.frameon": False})


def get_custom_colors(num_colors):
    """Return a deterministic repeating color palette."""
    palette = ["#33658A", "#86BBD8", "#2F4858", "#F6AE2D",
               "#9BC53D", "#55DDE0", "#F26419", "#758E4F"]
    return [palette[i % len(palette)] for i in range(num_colors)]


def load_data(filepath, h5_key):
    """Load an index-oriented CSV or an HDF5 DataFrame and validate its index."""
    log_message(f"Loading data from: {filepath}")
    extension = os.path.splitext(filepath)[1].lower()
    try:
        if extension == ".csv":
            frame = pd.read_csv(filepath, index_col=0)
        elif extension in (".h5", ".hdf5"):
            try:
                frame = pd.read_hdf(filepath, key=h5_key)
            except KeyError as exc:
                with pd.HDFStore(filepath, "r") as store:
                    keys = store.keys()
                raise KeyError(f"Key {h5_key!r} is absent from {filepath}; available: {keys}") from exc
        else:
            raise ValueError(f"Unsupported input extension {extension!r}; use CSV/H5/HDF5")
    except Exception:
        LOGGER.exception("Failed to load %s", filepath)
        raise
    if frame.index.has_duplicates:
        raise ValueError(f"Input index in {filepath} contains duplicate identifiers")
    index_text = pd.Series(frame.index, dtype="object").astype("string")
    if index_text.isna().any() or index_text.str.strip().eq("").any():
        raise ValueError(f"Input index in {filepath} contains null/blank identifiers")
    log_message(f"Successfully loaded {frame.shape[0]} rows and {frame.shape[1]} columns.")
    return frame


def sanitize_filename_component(value):
    """Return a filesystem-safe label while preserving already-safe labels."""
    import re
    text = str(value).strip()
    if re.fullmatch(r"[A-Za-z0-9._-]+", text):
        return text
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return safe or "label"


def make_model(args, seed, iterations=None):
    """Construct one deterministic CatBoost binary classifier."""
    return cb.CatBoostClassifier(
        iterations=args.iterations if iterations is None else iterations,
        learning_rate=args.learning_rate, depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg, auto_class_weights=args.auto_class_weights,
        thread_count=args.threads, verbose=0, random_seed=int(seed),
        allow_writing_files=False)


def _candidate_shap_arrays(values, expected, n_samples, n_features):
    """Enumerate plausible (phi, expected, descriptor) binary SHAP layouts."""
    expected_array = np.asarray(expected).reshape(-1)
    candidates = []

    def add(array, class_index, descriptor):
        arr = np.asarray(array)
        if arr.shape != (n_samples, n_features):
            return
        if expected_array.size == 1:
            ev = float(expected_array[0])
        elif class_index is not None and class_index < expected_array.size:
            ev = float(expected_array[class_index])
        else:
            return
        candidates.append((arr.astype(np.float64, copy=False), ev, descriptor))

    if isinstance(values, (list, tuple)):
        if len(values) == 1:
            add(values[0], None, "list[single_binary_output]")
        else:
            for class_index, array in enumerate(values):
                add(array, class_index, f"list[class={class_index}]")
        return candidates

    arr = np.asarray(values)
    if arr.ndim == 2:
        add(arr, None, "array[sample,feature]")
    elif arr.ndim == 3:
        # SHAP has emitted both sample-feature-output and output-sample-feature.
        if arr.shape[0] == n_samples and arr.shape[1] == n_features:
            for k in range(arr.shape[2]):
                add(arr[:, :, k], k, f"array[sample,feature,class={k}]")
        if arr.shape[0] == n_samples and arr.shape[2] == n_features:
            for k in range(arr.shape[1]):
                add(arr[:, k, :], k, f"array[sample,class={k},feature]")
        if arr.shape[1] == n_samples and arr.shape[2] == n_features:
            for k in range(arr.shape[0]):
                add(arr[k, :, :], k, f"array[class={k},sample,feature]")
    return candidates


def class1_tree_shap(model, X, context="model"):
    """Return class-1 TreeSHAP and verify exact raw-margin reconstruction.

    SHAP binary return layouts differ by version.  Every plausible class output is
    tested against CatBoost ``RawFormulaVal``.  A layout is accepted only when it
    reconstructs that class-1 margin; otherwise execution stops explicitly.
    """
    X_array = np.asarray(X, dtype=np.float64)
    raw_margin = np.asarray(model.predict(X_array, prediction_type="RawFormulaVal"),
                            dtype=np.float64).reshape(-1)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_array)
    candidates = _candidate_shap_arrays(values, explainer.expected_value,
                                         X_array.shape[0], X_array.shape[1])
    checks = []
    for phi, expected, descriptor in candidates:
        residual = raw_margin - (expected + phi.sum(axis=1))
        maximum = float(np.nanmax(np.abs(residual))) if residual.size else 0.0
        checks.append((maximum, phi, expected, descriptor, residual))
    tolerance = 1e-5 * max(1.0, float(np.nanmax(np.abs(raw_margin))))
    valid = [item for item in checks if np.isfinite(item[0]) and item[0] <= tolerance]
    if not valid:
        detail = ", ".join(f"{x[3]} max_residual={x[0]:.6g}" for x in checks) or "no compatible layout"
        raise RuntimeError(
            f"Unable to select class-1 SHAP output for {context}: {detail}. "
            "No SHAP output reconstructed CatBoost prediction_type='RawFormulaVal'.")
    # Prefer an explicitly labelled class-1 axis.  This matters for a degenerate
    # all-zero margin where both class outputs could satisfy the numerical test.
    def class_priority(item):
        descriptor = item[3]
        if "class=1" in descriptor:
            return 0
        if "class=" not in descriptor:
            return 1
        return 2
    valid.sort(key=lambda item: (class_priority(item), item[0], item[3]))
    best = valid[0]
    if "class=" in best[3] and "class=1" not in best[3]:
        raise RuntimeError(f"Only a non-class-1 SHAP axis reconstructed raw margins for {context}")
    # A class-0 output can reconstruct -margin, never +margin, so this is class 1.
    return best[1], best[2], raw_margin, best[4], best[3]


def safe_metrics(y_true, probability, prediction):
    """Compute binary metrics, returning NaN where a metric is undefined."""
    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1).astype(int)
    out = {
        "Accuracy": accuracy_score(y_true, prediction) if y_true.size else np.nan,
        "AUC": np.nan,
        "Precision": precision_score(y_true, prediction, zero_division=0) if y_true.size else np.nan,
        "Recall": recall_score(y_true, prediction, zero_division=0) if y_true.size else np.nan,
        "F1_Score": f1_score(y_true, prediction, zero_division=0) if y_true.size else np.nan,
        "Balanced_Accuracy": balanced_accuracy_score(y_true, prediction) if y_true.size and np.unique(y_true).size > 1 else np.nan,
    }
    if y_true.size and np.unique(y_true).size == 2:
        fpr, tpr, _ = roc_curve(y_true, probability)
        out["AUC"] = auc(fpr, tpr)
        out["fpr"], out["tpr"] = fpr, tpr
    else:
        out["fpr"], out["tpr"] = np.array([]), np.array([])
    return out


def stable_rank_frame(entity_ids, score, signed, score_name="mean_abs"):
    """Create deterministic score tables with tie-aware average ranks."""
    frame = pd.DataFrame({"entity_id": np.asarray(entity_ids, dtype=str),
                          score_name: np.asarray(score, dtype=float),
                          "signed_mean": np.asarray(signed, dtype=float)})
    frame = frame.sort_values([score_name, "entity_id"], ascending=[False, True],
                              kind="mergesort").reset_index(drop=True)
    frame["rank"] = frame[score_name].rank(method="average", ascending=False)
    return frame


def add_top_membership(frame, top_ks):
    """Add Top-K indicators, including every tie at the Kth score threshold."""
    result = frame.copy()
    n = len(result)
    for k in top_ks:
        if n == 0:
            result[f"top{k}"] = pd.Series(dtype=int)
            continue
        # Rows are score-sorted.  The rank of the Kth positional row is the
        # average rank shared by all entities tied at that score threshold.
        threshold_rank = result.iloc[min(k, n) - 1]["rank"]
        result[f"top{k}"] = (result["rank"] <= threshold_rank).astype(int)
    return result


def aggregate_families(tf_frame, mapping, top_ks):
    """Aggregate individual TF SHAP summaries into annotated families."""
    if mapping is None or mapping.empty or tf_frame.empty:
        return pd.DataFrame(columns=["entity_id", "sum_member_mean_abs", "mean_per_TF",
                                     "family_size", "signed_sum", "rank"] +
                                    [f"top{k}" for k in top_ks])
    work = tf_frame.merge(mapping[["geneID", "family"]], left_on="entity_id",
                          right_on="geneID", how="left")
    work["family"] = work["family"].fillna("Unannotated")
    # An unavailable/conflicting annotation is not a biological TF family.  Keep
    # those TFs in individual-level outputs, but never pool them into one
    # artificial and potentially dominant "Unannotated" family.
    work = work.loc[work["family"] != "Unannotated"].copy()
    if work.empty:
        return pd.DataFrame(columns=["entity_id", "sum_member_mean_abs", "mean_per_TF",
                                     "family_size", "signed_sum", "rank"] +
                                    [f"top{k}" for k in top_ks])
    grouped = work.groupby("family", sort=True).agg(
        sum_member_mean_abs=("mean_abs", "sum"), mean_per_TF=("mean_abs", "mean"),
        family_size=("entity_id", "nunique"), signed_sum=("signed_mean", "sum")).reset_index()
    grouped = grouped.rename(columns={"family": "entity_id"})
    grouped = grouped.sort_values(["sum_member_mean_abs", "entity_id"],
                                  ascending=[False, True], kind="mergesort").reset_index(drop=True)
    grouped["rank"] = grouped["sum_member_mean_abs"].rank(method="average", ascending=False)
    grouped["aggregation_definition"] = "aggregated individual TF TreeSHAP"
    return add_top_membership(grouped, top_ks)


def read_family_mapping(path, feature_columns):
    """Read geneID/family mapping, deduplicate, and mark conflicts explicitly."""
    columns = np.asarray(feature_columns, dtype=str)
    if path is None or not os.path.exists(path):
        status = pd.DataFrame({"geneID": columns, "family": "Unannotated",
                               "mapping_status": "Family_file_missing"})
        return status, "missing"
    try:
        # Project iTAK tables occur in both headerless two-column form and in
        # geneID/family-header form.  Reading without a header handles both;
        # the optional literal header row is removed explicitly below.
        table = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1],
                            names=["geneID", "family"], dtype=str,
                            comment="#")
    except Exception as exc:
        raise ValueError(f"Cannot parse two-column family file {path}: {exc}") from exc
    table = table[["geneID", "family"]].dropna()
    header_row = (table["geneID"].str.strip().str.lower().eq("geneid") &
                  table["family"].str.strip().str.lower().eq("family"))
    table = table.loc[~header_row].copy()
    table["geneID"] = table["geneID"].astype(str)
    table["family"] = table["family"].astype(str)
    table = table.drop_duplicates()
    by_gene = table.groupby("geneID")["family"].agg(lambda x: sorted(set(x)))
    rows = []
    for gene in columns:
        families = by_gene.get(gene, [])
        if len(families) == 1:
            rows.append((gene, families[0], "Mapped"))
        elif len(families) > 1:
            rows.append((gene, "Unannotated", "Conflict_multiple_families"))
        else:
            rows.append((gene, "Unannotated", "No_mapping"))
    return pd.DataFrame(rows, columns=["geneID", "family", "mapping_status"]), "loaded"


def pairwise_agreement(rank_tables, top_ks, level, score_kind):
    """Compare every pair of repeat rank vectors with tie-safe rank statistics."""
    rows = []
    for (run_a, a), (run_b, b) in itertools.combinations(sorted(rank_tables.items()), 2):
        merged = a[["entity_id", "rank"]].merge(b[["entity_id", "rank"]],
                                                  on="entity_id", suffixes=("_a", "_b"))
        if len(merged) >= 2:
            sp = spearmanr(merged["rank_a"], merged["rank_b"], nan_policy="omit").statistic
            kt = kendalltau(merged["rank_a"], merged["rank_b"], nan_policy="omit").statistic
        else:
            sp = kt = np.nan
        base = {"level": level, "score_kind": score_kind, "run_a": run_a,
                "run_b": run_b, "n_entities": len(merged), "spearman": sp,
                "kendall": kt}
        for k in top_ks:
            # Membership columns were built from Kth-score thresholds and hence
            # include all boundary ties; never truncate ties with nsmallest().
            col = f"top{k}"
            sa = set(a.loc[a[col].astype(bool), "entity_id"]) if col in a else set()
            sb = set(b.loc[b[col].astype(bool), "entity_id"]) if col in b else set()
            union = sa | sb
            base[f"top{k}_jaccard"] = len(sa & sb) / len(union) if union else np.nan
        rows.append(base)
    return pd.DataFrame(rows)


def percentile_interval(values):
    """Return finite median and empirical 2.5/97.5 percentiles."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan, np.nan, np.nan
    low, median, high = np.percentile(array, [2.5, 50, 97.5])
    return median, low, high


def summarize_rank_sources(run_frame, bootstrap_frame, top_ks):
    """Summarize repeated-run and true-refit bootstrap distributions separately."""
    entities = sorted(set(run_frame.get("entity_id", [])) | set(bootstrap_frame.get("entity_id", [])))
    rows = []
    for entity in entities:
        row = {"entity_id": entity}
        for prefix, frame in (("run", run_frame), ("bootstrap", bootstrap_frame)):
            subset = frame.loc[frame["entity_id"] == entity] if not frame.empty else frame
            for field in ("mean_abs", "rank"):
                median, low, high = percentile_interval(subset[field] if field in subset else [])
                row[f"{prefix}_{field}_median"] = median
                row[f"{prefix}_{field}_interval_low"] = low
                row[f"{prefix}_{field}_interval_high"] = high
                # Compatibility aliases for consumers of pre-review outputs.
                row[f"{prefix}_{field}_ci_low"] = low
                row[f"{prefix}_{field}_ci_high"] = high
            row[f"{prefix}_n"] = len(subset)
            for k in top_ks:
                col = f"top{k}"
                row[f"{prefix}_{col}_frequency"] = (subset[col].mean()
                                                       if col in subset and len(subset) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def family_summary_sources(run_frame, bootstrap_frame, top_ks):
    """Family equivalent of :func:`summarize_rank_sources`."""
    run = run_frame.rename(columns={"sum_member_mean_abs": "mean_abs"})
    boot = bootstrap_frame.rename(columns={"sum_member_mean_abs": "mean_abs"})
    return summarize_rank_sources(run, boot, top_ks)


def save_csv(frame, path):
    """Write a structured CSV even when it has zero rows."""
    frame.to_csv(path, index=False)
    log_message(f"Saved: {path}")


def plot_single_feature_performance(results, prefix, feature_name, colors):
    """Retain the original per-feature performance PDF name and semantics."""
    metrics = list(results["mean_scores"])
    means = [results["mean_scores"][x] for x in metrics]
    stds = [results["std_scores"][x] for x in metrics]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(metrics, means, yerr=stds, color=colors[0], capsize=8)
    ax.set(ylabel="Score", title=f"Model Performance Metrics: {feature_name}", ylim=(0, 1.05))
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", alpha=.4)
    fig.savefig(f"{prefix}_{feature_name}_Performance_Metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_performance_comparison(all_results, prefix, colors):
    """Retain the original cross-feature comparison output."""
    names = list(all_results)
    metrics = list(next(iter(all_results.values()))["mean_scores"])
    x = np.arange(len(metrics)); width = .8 / len(names)
    fig, ax = plt.subplots(figsize=(6 + 2.5 * len(names), 7))
    for i, name in enumerate(names):
        result = all_results[name]
        offset = width * (i - (len(names) - 1) / 2)
        ax.bar(x + offset, [result["mean_scores"][m] for m in metrics], width,
               yerr=[result["std_scores"][m] for m in metrics], label=name,
               color=colors[i], capsize=4)
    ax.set(xticks=x, xticklabels=metrics, ylabel="Scores", ylim=(0, 1.05),
           title="Model Performance Comparison")
    ax.legend(title="Feature Sets"); ax.grid(axis="y", linestyle="--", alpha=.4)
    fig.savefig(f"{prefix}_Performance_Comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(all_results, prefix, colors):
    """Retain legacy ROC coordinate CSVs and combined PDF."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (name, result) in enumerate(all_results.items()):
        usable = [fold for fold in result["fold_data"] if len(fold["fpr"])]
        if not usable:
            continue
        mean_fpr = np.linspace(0, 1, 100)
        tprs = [np.interp(mean_fpr, fold["fpr"], fold["tpr"]) for fold in usable]
        mean_tpr = np.mean(tprs, axis=0); mean_tpr[[0, -1]] = [0, 1]
        pd.DataFrame({"False_Positive_Rate": mean_fpr,
                      "True_Positive_Rate": mean_tpr}).to_csv(
                          f"{prefix}_{name}_ROC_curve_data.csv", index=False)
        aucs = [fold["AUC"] for fold in usable]
        ax.plot(mean_fpr, mean_tpr, color=colors[i], lw=2.5,
                label=f"{name} (AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random Chance")
    ax.set(xlim=(-.05, 1.05), ylim=(-.05, 1.05), xlabel="False Positive Rate",
           ylabel="True Positive Rate", title="Receiver Operating Characteristic (ROC) Curves")
    ax.legend(loc="lower right")
    fig.savefig(f"{prefix}_ROC_Curves.pdf", bbox_inches="tight")
    plt.close(fig)


def text_page(pdf, title, lines):
    """Add a robust text/status page to a multipage PDF."""
    fig, ax = plt.subplots(figsize=(11, 8.5)); ax.axis("off")
    ax.text(.03, .96, title, transform=ax.transAxes, va="top", fontsize=18, weight="bold")
    ax.text(.03, .88, "\n".join(str(x) for x in lines), transform=ax.transAxes,
            va="top", fontsize=10, linespacing=1.45, wrap=True)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def bar_interval_page(pdf, summary, prefix, title, top_n=20):
    """Plot score medians and empirical percentile intervals."""
    needed = f"{prefix}_mean_abs_median"
    usable = summary.dropna(subset=[needed]).copy() if needed in summary else pd.DataFrame()
    if usable.empty:
        text_page(pdf, title, ["No valid data available for this analysis."])
        return
    ordered = usable.sort_values([needed, "entity_id"], ascending=[False, True])
    threshold = ordered.iloc[min(top_n, len(ordered)) - 1][needed]
    usable = ordered.loc[ordered[needed] >= threshold].sort_values(needed)
    y = np.arange(len(usable)); center = usable[needed].to_numpy()
    low = usable[f"{prefix}_mean_abs_interval_low"].to_numpy(); high = usable[f"{prefix}_mean_abs_interval_high"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, max(5, .3 * len(usable))))
    ax.errorbar(center, y, xerr=np.vstack([center-low, high-center]), fmt="o", color="#33658A")
    ax.set(yticks=y, yticklabels=usable["entity_id"], xlabel="mean(abs(TreeSHAP))",
           title=title); ax.grid(axis="x", alpha=.25)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def rank_frequency_page(pdf, summary, prefix, top_ks, title, top_n=25):
    """Plot rank intervals and separate Top-K frequencies."""
    rank_col = f"{prefix}_rank_median"
    if rank_col not in summary or summary[rank_col].notna().sum() == 0:
        text_page(pdf, title, ["No valid rank distribution data available."])
        return
    ordered = summary.dropna(subset=[rank_col]).sort_values(
        [rank_col, "entity_id"], ascending=[True, True])
    threshold = ordered.iloc[min(top_n, len(ordered)) - 1][rank_col]
    data = ordered.loc[ordered[rank_col] <= threshold].copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, .3 * len(data))))
    y = np.arange(len(data)); center = data[rank_col].to_numpy()
    low = data[f"{prefix}_rank_interval_low"].to_numpy(); high = data[f"{prefix}_rank_interval_high"].to_numpy()
    ax1.errorbar(center, y, xerr=np.vstack([center-low, high-center]), fmt="o")
    ax1.set(yticks=y, yticklabels=data["entity_id"], xlabel="Rank (lower is better)",
            title="Empirical 2.5–97.5 percentile rank interval")
    width = .8 / max(1, len(top_ks))
    for j, k in enumerate(top_ks):
        col = f"{prefix}_top{k}_frequency"
        ax2.barh(y + (j-(len(top_ks)-1)/2)*width, data[col], height=width, label=f"Top {k}")
    ax2.set(yticks=y, yticklabels=[], xlim=(0, 1), xlabel="Frequency", title="Top-K membership")
    ax2.legend(); fig.suptitle(title)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def make_stability_pdf(path, manifest, performance, tf_summary,
                       tf_filtered_summary, family_summary,
                       family_filtered_summary, agreement, q3_summary,
                       q3_agreement, top_ks, test_mode):
    """Create a multipage report; every absent section receives a status page."""
    with PdfPages(path) as pdf:
        text_page(pdf, "TreeSHAP stability report", [
            f"Feature set: {manifest.get('feature_name')}",
            "TreeSHAP values are raw-margin contributions. A sign only indicates a push toward or away from encoded class 1; it has no standalone biological direction meaning.",
            "Dedicated performance CV and repeated OOF stability use separate configured splitters.",
            "Repeated stability CV: each gene is explained exactly once out-of-fold per repeat.",
            "Consensus: elementwise mean across repeated OOF matrices; it is not a single-model explanation and excludes bootstrap models.",
            "Bootstrap: class-stratified row resampling followed by a true model refit; fixed complete X is explained for comparable ranks.",
            "Family scores aggregate individual TF SHAP (sum member mean-absolute is the primary family rank).",
            "Preprocessing: raw features are used for every CV fold, refit, and final model (no full-data scaling).",
            f"Status: {manifest.get('stability_status')}; family: {manifest.get('family_status')}",
        ])
        if performance.empty:
            text_page(pdf, "CV and OOB performance", ["No performance records available."])
        else:
            fold = performance[performance["record_type"].isin(
                ["performance_CV_fold", "stability_CV_fold"])].copy()
            fig, ax = plt.subplots(figsize=(11, 6))
            if fold.empty:
                ax.axis("off"); ax.text(.1, .5, "No CV fold performance available.")
            else:
                sns.boxplot(data=fold, x="metric", y="value", ax=ax, color="#86BBD8")
                if fold["repeat"].notna().any():
                    sns.stripplot(data=fold, x="metric", y="value", hue="repeat", ax=ax,
                                  palette="viridis", dodge=False, size=4)
                else:
                    sns.stripplot(data=fold, x="metric", y="value", ax=ax,
                                  color="#33658A", size=4)
                ax.tick_params(axis="x", rotation=25)
                ax.set(title="Dedicated performance and repeated-stability CV folds",
                       ylim=(-.05, 1.05))
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        bar_interval_page(pdf, tf_summary, "run", "TF raw OOF score: empirical repeated-run percentile intervals")
        bar_interval_page(pdf, tf_filtered_summary, "run", "TF zero-input-filtered OOF score: empirical repeated-run percentile intervals")
        bar_interval_page(pdf, tf_summary, "bootstrap", "TF raw score: empirical refit-stability percentile intervals")
        rank_frequency_page(pdf, tf_summary, "run", top_ks, "TF repeated-run ranks and Top-K frequencies")
        rank_frequency_page(pdf, tf_summary, "bootstrap", top_ks, "TF bootstrap ranks and Top-K frequencies")
        if family_summary.empty:
            text_page(pdf, "Family stability", ["No family stability data. See mapping status CSV."])
        else:
            bar_interval_page(pdf, family_summary, "run", "Family raw empirical repeated-run percentile intervals")
            rank_frequency_page(pdf, family_summary, "bootstrap", top_ks,
                                "Family raw bootstrap ranks and Top-K frequencies")
        if family_filtered_summary.empty:
            text_page(pdf, "Filtered family stability", [
                "No zero-input-filtered family stability data. See mapping status CSV."])
        else:
            bar_interval_page(pdf, family_filtered_summary, "run",
                              "Family zero-input-filtered empirical repeated-run percentile intervals")
            rank_frequency_page(
                pdf, family_filtered_summary, "bootstrap", top_ks,
                "Family zero-input-filtered bootstrap ranks and Top-K frequencies")
        if agreement.empty:
            text_page(pdf, "Run agreement", ["Pairwise agreement requires at least two successful repeats (R=1 has no pairs)."])
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            sns.boxplot(data=agreement, x="level", y="spearman", ax=axes[0], color="#33658A")
            sns.boxplot(data=agreement, x="level", y="kendall", ax=axes[1], color="#F6AE2D")
            axes[0].set(title="Pairwise Spearman", ylim=(-1.05, 1.05)); axes[1].set(title="Pairwise Kendall", ylim=(-1.05, 1.05))
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        text_page(pdf, "Additivity diagnostics", [
            f"Raw maximum |residual|: {q3_summary.get('raw_residual_abs_max', np.nan)}",
            f"Filtered identity maximum error: {q3_summary.get('filtered_identity_error_abs_max', np.nan)}",
            "raw residual = raw margin - (expected + sum raw phi)",
            "filtered residual = raw margin - (expected + sum filtered phi)",
            "removed contribution = sum raw phi - sum filtered phi",
            "Automatically checked: filtered residual ≈ raw residual + removed contribution.",
        ])
        if q3_agreement.empty:
            text_page(pdf, "Raw versus filtered ranks", ["No rank-comparison data available."])
        else:
            text_page(pdf, "Raw versus filtered ranks", [q3_agreement.to_string(index=False)])
        if test_mode:
            text_page(pdf, "Test-mode disclosure", ["stability_test_mode was enabled: stability model counts and report detail may be reduced; legacy final-model semantics remain unchanged."])
    log_message(f"Saved multipage stability report: {path}")


def package_version(name):
    """Return package version without introducing a dependency."""
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def process_feature(args, feature_path, deg_data, label_encoder):
    """Run legacy final-model and all requested stability analyses for one feature."""
    start = time.time()
    feature_name = os.path.splitext(os.path.basename(feature_path))[0]
    feature_data = load_data(feature_path, args.h5_key)
    merged = feature_data.join(deg_data, how="inner")
    if merged.empty:
        raise ValueError(f"Feature join for {feature_name} has no common gene identifiers")
    if merged.index.has_duplicates:
        raise ValueError(f"Merged gene index for {feature_name} contains duplicates")
    X = merged.drop(columns=["DE_Label", "DE_Label_Encoded"])
    if X.shape[1] < 1:
        raise ValueError(f"Feature matrix {feature_name} must contain at least one feature")
    if X.columns.duplicated().any():
        raise ValueError(f"Feature names for {feature_name} are not unique")
    X_values = X.to_numpy(dtype=float)
    if not np.isfinite(X_values).all():
        raise ValueError(f"Feature matrix {feature_name} contains NaN/inf")
    y = merged["DE_Label_Encoded"].to_numpy(dtype=int)
    classes = np.unique(y)
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError(f"Feature join for {feature_name} must retain both classes; found {classes.tolist()}")
    class_counts = np.bincount(y, minlength=2)
    required_splits = args.performance_splits
    if not args.skip_stability:
        required_splits = max(required_splits, args.stability_splits)
    if class_counts.min() < required_splits:
        raise ValueError(
            f"Each joined class needs at least {required_splits} rows for requested CV splits; "
            f"counts={class_counts.tolist()}")

    genes = merged.index.astype(str).to_numpy(); tfs = X.columns.astype(str).to_numpy()
    zero_mask = X_values == 0
    top_ks = args.stability_top_k
    requested_r = args.stability_repeats
    requested_b = args.stability_bootstraps
    effective_r = min(requested_r, 2) if args.stability_test_mode else requested_r
    effective_b = min(requested_b, 2) if args.stability_test_mode else requested_b
    stability_iterations = min(args.iterations, 25) if args.stability_test_mode else args.iterations
    if args.skip_stability:
        effective_r = effective_b = 0

    auto_family = os.path.join(os.path.dirname(os.path.abspath(feature_path)), "0_At_TF_list_itak.txt")
    family_path = args.family_file if args.family_file is not None else auto_family
    mapping, family_status = read_family_mapping(family_path, tfs)
    base = f"{args.out_prefix}_{feature_name}_stability"
    save_csv(mapping, f"{base}_family_mapping.csv")

    run_tf_rows, run_filtered_rows = [], []
    family_run_rows, family_filtered_run_rows = [], []
    performance_long = []
    rank_raw_tables, rank_filtered_tables = {}, {}
    family_rank_tables, family_filtered_rank_tables = {}, {}
    consensus_raw = np.zeros((len(genes), len(tfs)), dtype=np.float32)
    consensus_filtered = np.zeros_like(consensus_raw)
    margin_runs = np.zeros((len(genes), effective_r), dtype=np.float32)
    expected_runs = np.zeros((len(genes), effective_r), dtype=np.float32)
    cv_first_fold_data = []

    for repeat in range(effective_r):
        split_seed = 42 if repeat == 0 else args.stability_seed + repeat
        cv = StratifiedKFold(n_splits=args.stability_splits, shuffle=True,
                            random_state=split_seed)
        oof = np.empty((len(genes), len(tfs)), dtype=np.float32)
        oof_margin = np.empty(len(genes), dtype=np.float32)
        oof_expected = np.empty(len(genes), dtype=np.float32)
        seen = np.zeros(len(genes), dtype=np.int16)
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_values, y)):
            seed = (args.stability_seed + repeat * args.stability_splits + fold)
            model = make_model(args, seed, stability_iterations)
            model.fit(X_values[train_idx], y[train_idx])
            probability = model.predict_proba(X_values[val_idx])[:, 1]
            prediction = np.asarray(model.predict(X_values[val_idx])).reshape(-1)
            metrics = safe_metrics(y[val_idx], probability, prediction)
            for metric in ("Accuracy", "AUC", "Precision", "Recall", "F1_Score", "Balanced_Accuracy"):
                performance_long.append({"record_type": "stability_CV_fold", "repeat": repeat + 1,
                                         "fold_or_bootstrap": fold + 1, "seed": seed,
                                         "metric": metric, "value": metrics[metric], "status": "OK"})
            phi, expected, raw_margin, residual, layout = class1_tree_shap(
                model, X_values[val_idx], f"{feature_name} repeat {repeat+1} fold {fold+1}")
            oof[val_idx] = phi.astype(np.float32)
            oof_margin[val_idx] = raw_margin.astype(np.float32)
            oof_expected[val_idx] = np.float32(expected)
            seen[val_idx] += 1
            del phi, model
        if not np.all(seen == 1):
            raise RuntimeError(f"OOF invariant failed in repeat {repeat+1}: counts {np.unique(seen, return_counts=True)}")
        filtered = oof * (~zero_mask)
        consensus_raw += oof / np.float32(effective_r)
        consensus_filtered += filtered / np.float32(effective_r)
        margin_runs[:, repeat] = oof_margin; expected_runs[:, repeat] = oof_expected
        raw_rank = add_top_membership(stable_rank_frame(tfs, np.mean(np.abs(oof), axis=0),
                                                        np.mean(oof, axis=0)), top_ks)
        filtered_rank = add_top_membership(stable_rank_frame(tfs, np.mean(np.abs(filtered), axis=0),
                                                             np.mean(filtered, axis=0)), top_ks)
        raw_rank.insert(0, "repeat", repeat + 1); filtered_rank.insert(0, "repeat", repeat + 1)
        run_tf_rows.append(raw_rank); run_filtered_rows.append(filtered_rank)
        rank_raw_tables[repeat + 1] = raw_rank
        rank_filtered_tables[repeat + 1] = filtered_rank
        fam = aggregate_families(raw_rank, mapping, top_ks)
        if not fam.empty:
            fam.insert(0, "repeat", repeat + 1); family_run_rows.append(fam)
            family_rank_tables[repeat + 1] = fam
        fam_filtered = aggregate_families(filtered_rank, mapping, top_ks)
        if not fam_filtered.empty:
            fam_filtered.insert(0, "repeat", repeat + 1)
            family_filtered_run_rows.append(fam_filtered)
            family_filtered_rank_tables[repeat + 1] = fam_filtered
        del oof, filtered

    # Performance estimation is deliberately independent of repeated OOF
    # stability: it always uses its own splitter and full requested iterations.
    performance_cv = StratifiedKFold(n_splits=args.performance_splits, shuffle=True,
                                    random_state=42)
    for fold, (train_idx, val_idx) in enumerate(performance_cv.split(X_values, y)):
        seed = 42 + fold
        model = make_model(args, seed, args.iterations)
        model.fit(X_values[train_idx], y[train_idx])
        probability = model.predict_proba(X_values[val_idx])[:, 1]
        prediction = np.asarray(model.predict(X_values[val_idx])).reshape(-1)
        metrics = safe_metrics(y[val_idx], probability, prediction)
        cv_first_fold_data.append({"Feature_Set": feature_name, "Fold": fold + 1,
            **{m: metrics[m] for m in ("Accuracy", "AUC", "Precision", "Recall", "F1_Score", "Balanced_Accuracy")},
            "fpr": metrics["fpr"], "tpr": metrics["tpr"]})
        for metric in ("Accuracy", "AUC", "Precision", "Recall", "F1_Score", "Balanced_Accuracy"):
            performance_long.append({"record_type": "performance_CV_fold", "repeat": np.nan,
                                     "fold_or_bootstrap": fold + 1, "seed": seed,
                                     "metric": metric, "value": metrics[metric],
                                     "status": "OK"})
        del model

    tf_run = pd.concat(run_tf_rows, ignore_index=True) if run_tf_rows else pd.DataFrame(
        columns=["repeat", "entity_id", "mean_abs", "signed_mean", "rank"] + [f"top{k}" for k in top_ks])
    tf_filtered_run = pd.concat(run_filtered_rows, ignore_index=True) if run_filtered_rows else pd.DataFrame(columns=tf_run.columns)
    family_run = pd.concat(family_run_rows, ignore_index=True) if family_run_rows else pd.DataFrame()
    family_filtered_run = (pd.concat(family_filtered_run_rows, ignore_index=True)
                           if family_filtered_run_rows else pd.DataFrame())

    # True model-refit bootstrap: train on NumPy duplicate rows; explain fixed full X.
    bootstrap_tf_rows, bootstrap_filtered_rows = [], []
    bootstrap_family_rows, bootstrap_filtered_family_rows = [], []
    for boot in range(effective_b):
        seed = args.stability_seed + 100000 + boot
        rng = np.random.default_rng(seed)
        sampled_parts = [rng.choice(np.flatnonzero(y == cls), size=np.sum(y == cls), replace=True)
                         for cls in (0, 1)]
        sampled = np.concatenate(sampled_parts); rng.shuffle(sampled)
        model = make_model(args, seed, stability_iterations)
        model.fit(np.asarray(X_values[sampled]), np.asarray(y[sampled]))
        inbag = np.zeros(len(y), dtype=bool); inbag[np.unique(sampled)] = True
        oob = np.flatnonzero(~inbag)
        if oob.size and np.unique(y[oob]).size == 2:
            prob = model.predict_proba(X_values[oob])[:, 1]
            pred = np.asarray(model.predict(X_values[oob])).reshape(-1)
            metrics = safe_metrics(y[oob], prob, pred); status = "OK"
        else:
            metrics = {m: np.nan for m in ("Accuracy", "AUC", "Precision", "Recall", "F1_Score", "Balanced_Accuracy")}
            status = "NA_OOB_missing_class" if oob.size else "NA_no_OOB_rows"
        for metric in metrics:
            if metric in ("fpr", "tpr"): continue
            performance_long.append({"record_type": "bootstrap_OOB", "repeat": np.nan,
                                     "fold_or_bootstrap": boot + 1, "seed": seed,
                                     "metric": metric, "value": metrics[metric], "status": status})
        phi, expected, raw_margin, residual, layout = class1_tree_shap(
            model, X_values, f"{feature_name} bootstrap {boot+1}")
        rank = add_top_membership(stable_rank_frame(tfs, np.mean(np.abs(phi), axis=0),
                                                    np.mean(phi, axis=0)), top_ks)
        filtered_phi = phi * (~zero_mask)
        filtered_rank = add_top_membership(
            stable_rank_frame(tfs, np.mean(np.abs(filtered_phi), axis=0),
                              np.mean(filtered_phi, axis=0)), top_ks)
        rank.insert(0, "bootstrap", boot + 1); rank.insert(1, "seed", seed)
        filtered_rank.insert(0, "bootstrap", boot + 1)
        filtered_rank.insert(1, "seed", seed)
        bootstrap_tf_rows.append(rank)
        bootstrap_filtered_rows.append(filtered_rank)
        fam = aggregate_families(rank, mapping, top_ks)
        if not fam.empty:
            fam.insert(0, "bootstrap", boot + 1); fam.insert(1, "seed", seed)
            bootstrap_family_rows.append(fam)
        fam_filtered = aggregate_families(filtered_rank, mapping, top_ks)
        if not fam_filtered.empty:
            fam_filtered.insert(0, "bootstrap", boot + 1)
            fam_filtered.insert(1, "seed", seed)
            bootstrap_filtered_family_rows.append(fam_filtered)
        del phi, filtered_phi, model, raw_margin, residual

    empty_boot_cols = (["bootstrap", "seed", "entity_id", "mean_abs", "signed_mean", "rank"] +
                       [f"top{k}" for k in top_ks])
    tf_boot = (pd.concat(bootstrap_tf_rows, ignore_index=True) if bootstrap_tf_rows
               else pd.DataFrame(columns=empty_boot_cols))
    tf_filtered_boot = (pd.concat(bootstrap_filtered_rows, ignore_index=True)
                        if bootstrap_filtered_rows else pd.DataFrame(columns=empty_boot_cols))
    family_boot = pd.concat(bootstrap_family_rows, ignore_index=True) if bootstrap_family_rows else pd.DataFrame()
    family_filtered_boot = (pd.concat(bootstrap_filtered_family_rows, ignore_index=True)
                            if bootstrap_filtered_family_rows else pd.DataFrame())
    tf_summary = summarize_rank_sources(tf_run, tf_boot, top_ks)
    tf_summary.insert(1 if "entity_id" in tf_summary else 0, "score_kind", "raw")
    tf_filtered_summary = summarize_rank_sources(tf_filtered_run, tf_filtered_boot, top_ks)
    tf_filtered_summary.insert(1 if "entity_id" in tf_filtered_summary else 0,
                               "score_kind", "zero_input_filtered")
    tf_summary_export = pd.concat([tf_summary, tf_filtered_summary], ignore_index=True,
                                  sort=False)
    family_summary = family_summary_sources(family_run, family_boot, top_ks)
    family_summary.insert(1 if "entity_id" in family_summary else 0, "score_kind", "raw")
    family_filtered_summary = family_summary_sources(
        family_filtered_run, family_filtered_boot, top_ks)
    family_filtered_summary.insert(
        1 if "entity_id" in family_filtered_summary else 0,
        "score_kind", "zero_input_filtered")
    family_summary_export = pd.concat(
        [family_summary, family_filtered_summary], ignore_index=True, sort=False)

    agreement_parts = [pairwise_agreement(rank_raw_tables, top_ks, "TF", "raw"),
                       pairwise_agreement(rank_filtered_tables, top_ks, "TF", "filtered"),
                       pairwise_agreement(family_rank_tables, top_ks, "family", "aggregated_raw"),
                       pairwise_agreement(family_filtered_rank_tables, top_ks, "family", "aggregated_filtered")]
    agreement = pd.concat([x for x in agreement_parts if not x.empty], ignore_index=True) if any(not x.empty for x in agreement_parts) else pd.DataFrame(
        columns=["level", "score_kind", "run_a", "run_b", "n_entities", "spearman", "kendall"] + [f"top{k}_jaccard" for k in top_ks])

    # Legacy-compatible final model and final raw/filtered SHAP outputs.
    final_model = make_model(args, 42, args.iterations)
    final_model.fit(X_values, y)
    final_phi, final_expected, final_margin, final_residual, final_layout = class1_tree_shap(
        final_model, X_values, f"{feature_name} legacy final model")
    final_filtered = final_phi * (~zero_mask)
    label0, label1 = label_encoder.classes_[0], label_encoder.classes_[1]
    safe_label0 = sanitize_filename_component(label0)
    safe_label1 = sanitize_filename_component(label1)
    raw_name = (f"{args.out_prefix}_{feature_name}_SHAP_raw_values_exp_{final_expected:.4f}_"
                f"pos_is_{safe_label1}_neg_is_{safe_label0}.csv")
    filtered_name = (f"{args.out_prefix}_{feature_name}_SHAP_filtered_values_exp_{final_expected:.4f}_"
                     f"pos_is_{safe_label1}_neg_is_{safe_label0}.csv")
    pd.DataFrame(final_phi, index=merged.index, columns=X.columns).to_csv(raw_name)
    pd.DataFrame(final_filtered, index=merged.index, columns=X.columns).to_csv(filtered_name)
    log_message(f"Saved legacy raw/filtered SHAP: {raw_name}; {filtered_name}")
    plt.figure()
    shap.summary_plot(final_phi, pd.DataFrame(X_values, index=merged.index, columns=X.columns),
                      show=False, max_display=20, plot_type="dot")
    plt.title(f"SHAP Feature Importance for {feature_name}\n(positive pushes toward encoded class 1: {label1})",
              fontsize=13, pad=20)
    plt.savefig(f"{args.out_prefix}_{feature_name}_SHAP_summary_top20.pdf", bbox_inches="tight")
    plt.close()

    # Q3 exact raw/filtered diagnostics and rank comparison.
    raw_sum = final_phi.sum(axis=1); filtered_sum = final_filtered.sum(axis=1)
    raw_resid = final_margin - (final_expected + raw_sum)
    filtered_resid = final_margin - (final_expected + filtered_sum)
    removed = raw_sum - filtered_sum
    identity_error = filtered_resid - (raw_resid + removed)
    q3_diag = pd.DataFrame({"geneID": genes, "encoded_label": y,
        "raw_margin": final_margin, "expected_value": final_expected,
        "raw_phi_sum": raw_sum, "filtered_phi_sum": filtered_sum,
        "raw_residual": raw_resid, "filtered_residual": filtered_resid,
        "removed_contribution": removed, "filtered_identity_error": identity_error})
    identity_tol = 1e-8 * max(1.0, float(np.max(np.abs(final_margin))))
    identity_ok = bool(np.max(np.abs(identity_error)) <= identity_tol)
    if not identity_ok:
        raise RuntimeError("Q3 filtered residual identity check failed")
    q3_summary_dict = {"n_genes": len(genes), "shap_layout": final_layout,
        "expected_value": final_expected, "raw_residual_abs_max": float(np.max(np.abs(raw_resid))),
        "raw_residual_abs_mean": float(np.mean(np.abs(raw_resid))),
        "filtered_residual_abs_max": float(np.max(np.abs(filtered_resid))),
        "removed_contribution_abs_mean": float(np.mean(np.abs(removed))),
        "filtered_identity_error_abs_max": float(np.max(np.abs(identity_error))),
        "filtered_identity_check": identity_ok, "identity_tolerance": identity_tol}
    q3_summary = pd.DataFrame([q3_summary_dict])
    q3_raw_rank = add_top_membership(
        stable_rank_frame(tfs, np.mean(np.abs(final_phi), axis=0),
                          np.mean(final_phi, axis=0)), top_ks)
    q3_filtered_rank = add_top_membership(
        stable_rank_frame(tfs, np.mean(np.abs(final_filtered), axis=0),
                          np.mean(final_filtered, axis=0)), top_ks)
    q3_rank = q3_raw_rank.merge(q3_filtered_rank, on="entity_id", suffixes=("_raw", "_filtered"))
    q3_rank["rank_change_filtered_minus_raw"] = q3_rank["rank_filtered"] - q3_rank["rank_raw"]
    q3_agree = {"n_entities": len(q3_rank),
        "spearman": spearmanr(q3_rank["rank_raw"], q3_rank["rank_filtered"]).statistic if len(q3_rank)>1 else np.nan,
        "kendall": kendalltau(q3_rank["rank_raw"], q3_rank["rank_filtered"]).statistic if len(q3_rank)>1 else np.nan}
    for k in top_ks:
        a = set(q3_rank.loc[q3_rank[f"top{k}_raw"].astype(bool), "entity_id"])
        b = set(q3_rank.loc[q3_rank[f"top{k}_filtered"].astype(bool), "entity_id"])
        q3_agree[f"top{k}_jaccard"] = len(a & b) / len(a | b) if a | b else np.nan
    q3_agreement = pd.DataFrame([q3_agree])

    # Consensus is repeated OOF only, never bootstrap. Empty-but-explicit when skipped.
    if effective_r:
        mean_margin = margin_runs.mean(axis=1); mean_expected = expected_runs.mean(axis=1)
        c_raw_sum = consensus_raw.sum(axis=1); c_filtered_sum = consensus_filtered.sum(axis=1)
        consensus_meta = pd.DataFrame({"geneID": genes, "encoded_label": y,
            "original_label": merged["DE_Label"].astype(str).to_numpy(),
            "mean_raw_margin": mean_margin, "mean_expected_value": mean_expected,
            "consensus_raw_phi_sum": c_raw_sum, "consensus_filtered_phi_sum": c_filtered_sum,
            "raw_residual": mean_margin-(mean_expected+c_raw_sum),
            "filtered_residual": mean_margin-(mean_expected+c_filtered_sum),
            "removed_contribution": c_raw_sum-c_filtered_sum,
            "n_oof_repeats": effective_r,
            "consensus_definition": "elementwise mean of repeated OOF TreeSHAP; excludes bootstrap"})
    else:
        consensus_meta = pd.DataFrame(columns=["geneID", "encoded_label", "original_label",
            "mean_raw_margin", "mean_expected_value", "consensus_raw_phi_sum",
            "consensus_filtered_phi_sum", "raw_residual", "filtered_residual",
            "removed_contribution", "n_oof_repeats", "consensus_definition"])

    performance = pd.DataFrame(performance_long, columns=["record_type", "repeat", "fold_or_bootstrap", "seed", "metric", "value", "status"])
    save_csv(performance, f"{base}_run_performance.csv")
    tf_run_export = pd.concat([
        tf_run.assign(score_kind="raw"),
        tf_filtered_run.assign(score_kind="zero_input_filtered")
    ], ignore_index=True, sort=False)
    save_csv(tf_run_export, f"{base}_TF_run_ranks.csv")
    save_csv(tf_filtered_run, f"{base}_TF_filtered_run_ranks.csv")
    tf_boot_export = pd.concat([
        tf_boot.assign(score_kind="raw"),
        tf_filtered_boot.assign(score_kind="zero_input_filtered")
    ], ignore_index=True, sort=False)
    save_csv(tf_boot_export, f"{base}_TF_bootstrap_ranks.csv")
    save_csv(tf_filtered_boot, f"{base}_TF_filtered_bootstrap_ranks.csv")
    save_csv(tf_summary_export, f"{base}_TF_summary.csv")
    family_run_export = pd.concat([
        family_run.assign(score_kind="raw"),
        family_filtered_run.assign(score_kind="zero_input_filtered")
    ], ignore_index=True, sort=False)
    family_boot_export = pd.concat([
        family_boot.assign(score_kind="raw"),
        family_filtered_boot.assign(score_kind="zero_input_filtered")
    ], ignore_index=True, sort=False)
    save_csv(family_run_export, f"{base}_family_run_ranks.csv")
    save_csv(family_boot_export, f"{base}_family_bootstrap_ranks.csv")
    save_csv(family_summary_export, f"{base}_family_summary.csv")
    save_csv(agreement, f"{base}_pairwise_agreement.csv")
    save_csv(q3_diag, f"{base}_Q3_additivity_diagnostics.csv")
    save_csv(q3_summary, f"{base}_Q3_additivity_summary.csv")
    save_csv(q3_rank, f"{base}_Q3_rank_comparison.csv")
    save_csv(q3_agreement, f"{base}_Q3_rank_agreement.csv")
    save_csv(consensus_meta, f"{base}_consensus_metadata.csv")
    if effective_r:
        pd.DataFrame(consensus_raw, index=merged.index, columns=X.columns).to_csv(
            f"{base}_consensus_raw_SHAP.csv")
        pd.DataFrame(consensus_filtered, index=merged.index, columns=X.columns).to_csv(
            f"{base}_consensus_filtered_SHAP.csv")
    else:
        pd.DataFrame(columns=X.columns).to_csv(f"{base}_consensus_raw_SHAP.csv")
        pd.DataFrame(columns=X.columns).to_csv(f"{base}_consensus_filtered_SHAP.csv")

    elapsed = time.time() - start
    manifest = {
        "feature_name": feature_name, "feature_path": os.path.abspath(feature_path),
        "family_path": os.path.abspath(family_path), "family_status": family_status,
        "stability_status": "skipped_by_user" if args.skip_stability else "complete",
        "preprocessing": "raw features; no scaling",
        "legacy_scaling": "disabled (raw features used)",
        "n_genes": len(genes), "n_TFs": len(tfs), "class_0_count": int(class_counts[0]),
        "class_1_count": int(class_counts[1]), "requested_repeats": requested_r,
        "effective_repeats": effective_r, "requested_bootstraps": requested_b,
        "effective_bootstraps": effective_b,
        "performance_splits": args.performance_splits,
        "stability_splits": args.stability_splits,
        "n_splits": args.n_splits,
        "iterations": args.iterations, "stability_effective_iterations": stability_iterations,
        "learning_rate": args.learning_rate, "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg, "auto_class_weights": args.auto_class_weights,
        "threads": args.threads, "h5_key": args.h5_key,
        "stability_seed": args.stability_seed,
        "seed_scheme": "performance split=42/folds=42+fold; stability splits=42 then stability_seed+repeat; stability folds=stability_seed+repeat*stability_splits+fold; bootstrap=stability_seed+100000+b",
        "top_k": json.dumps(top_ks), "skip_stability": args.skip_stability,
        "stability_test_mode": args.stability_test_mode,
        "interval_definition": "empirical 2.5th/97.5th percentile stability interval",
        "repeat_models": effective_r * args.stability_splits,
        "performance_models": args.performance_splits,
        "bootstrap_models": effective_b,
        "legacy_cv_only_models": 0,
        "final_models": 1,
        "total_models": (effective_r * args.stability_splits +
                         args.performance_splits + effective_b + 1),
        "elapsed_seconds": elapsed, "python_version": platform.python_version(),
        "platform": platform.platform(), "numpy_version": np.__version__,
        "pandas_version": pd.__version__, "sklearn_version": sklearn.__version__,
        "scipy_version": scipy.__version__, "catboost_version": cb.__version__,
        "shap_version": shap.__version__, "matplotlib_version": package_version("matplotlib"),
        "seaborn_version": sns.__version__, "final_shap_layout": final_layout,
        "consensus_includes_bootstrap": False,
        "encoded_class_0": str(label0), "encoded_class_1": str(label1),
        "filename_class_0": safe_label0, "filename_class_1": safe_label1,
        "sign_interpretation": "pushes toward/away from encoded class 1 only",
        "command": " ".join(sys.argv),
    }
    manifest_df = pd.DataFrame({"key": list(manifest), "value": [manifest[k] for k in manifest]})
    save_csv(manifest_df, f"{base}_manifest_config.csv")
    make_stability_pdf(f"{args.out_prefix}_{feature_name}_stability_report.pdf", manifest,
                       performance, tf_summary, tf_filtered_summary, family_summary,
                       family_filtered_summary, agreement, q3_summary_dict,
                       q3_agreement, top_ks, args.stability_test_mode)

    fold_df = pd.DataFrame(cv_first_fold_data)
    if fold_df.empty:
        legacy_result = None
    else:
        metric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1_Score", "Balanced_Accuracy"]
        legacy_result = {"fold_data": cv_first_fold_data,
                         "mean_scores": fold_df[metric_columns].mean().to_dict(),
                         "std_scores": fold_df[metric_columns].std().to_dict()}
    return {"feature_name": feature_name, "legacy_result": legacy_result,
            "legacy_folds": fold_df, "manifest": manifest}


def parse_args(argv=None):
    """Define legacy and stability CLI arguments."""
    parser = argparse.ArgumentParser(description="Train CatBoost and explain with stability-validated TreeSHAP.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out_prefix", type=str, required=True, help="Prefix for all output files.")
    parser.add_argument("--TF_features", type=str, nargs="+", required=True, help="One or more CSV/H5 feature files.")
    parser.add_argument("--DEGs", type=str, required=True, help="DEG labels, indexed by gene.")
    parser.add_argument("--h5_key", type=str, default="/regulons")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--l2_leaf_reg", type=float, default=3.0)
    parser.add_argument("--auto_class_weights", type=str, default="Balanced", choices=["None", "Balanced"])
    parser.add_argument("--performance_splits", type=int, default=5,
                        help="Folds used only for dedicated performance estimation.")
    parser.add_argument("--stability_splits", type=int, default=10,
                        help="Folds used only within each repeated OOF stability run.")
    parser.add_argument("--n_splits", type=int, default=None,
                        help="Deprecated alias; when supplied, sets both split counts.")
    parser.add_argument("--stability_repeats", type=int, default=5)
    parser.add_argument("--stability_bootstraps", type=int, default=30)
    parser.add_argument("--stability_seed", type=int, default=42)
    parser.add_argument("--stability_top_k", type=int, nargs="+", default=[10, 15, 20])
    parser.add_argument("--family_file", type=str, default=None)
    parser.add_argument("--skip_stability", action="store_true")
    parser.add_argument("--stability_test_mode", action="store_true")
    args = parser.parse_args(argv)
    if args.n_splits is not None:
        if args.n_splits < 2:
            parser.error("--n_splits must be >=2")
        args.performance_splits = args.n_splits
        args.stability_splits = args.n_splits
        warnings.warn("--n_splits is deprecated; use --performance_splits and "
                      "--stability_splits", DeprecationWarning, stacklevel=2)
    if args.performance_splits < 2 or args.stability_splits < 2:
        parser.error("--performance_splits and --stability_splits must each be >=2")
    if args.stability_repeats < 1:
        parser.error("--stability_repeats must be >=1")
    if args.stability_bootstraps < 0:
        parser.error("--stability_bootstraps must be >=0")
    if args.iterations < 1 or args.threads < 1:
        parser.error("--iterations and --threads must be >=1")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning_rate must be finite and >0")
    if not np.isfinite(args.l2_leaf_reg) or args.l2_leaf_reg < 0:
        parser.error("--l2_leaf_reg must be finite and >=0")
    if not 1 <= args.depth <= 16:
        parser.error("--depth must be in CatBoost's supported range 1..16")
    if any(k < 1 for k in args.stability_top_k):
        parser.error("all --stability_top_k values must be positive")
    if len(args.TF_features) < 1:
        parser.error("at least one --TF_features file is required")
    for option, path in [("--DEGs", args.DEGs)] + [("--TF_features", p) for p in args.TF_features]:
        if not os.path.isfile(path):
            parser.error(f"{option} path is not a readable file: {path}")
    if args.family_file is not None and not os.path.isfile(args.family_file):
        parser.error(f"--family_file path is not a readable file: {args.family_file}")
    feature_names = [os.path.splitext(os.path.basename(path))[0]
                     for path in args.TF_features]
    duplicates = sorted({name for name in feature_names if feature_names.count(name) > 1})
    if duplicates:
        parser.error("duplicate derived feature basenames would overwrite outputs: " +
                     ", ".join(duplicates))
    args.stability_top_k = sorted(set(args.stability_top_k))
    if args.auto_class_weights == "None":
        args.auto_class_weights = None
    return args


def main(argv=None):
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    warnings.filterwarnings("default")
    args = parse_args(argv)
    set_plot_style()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    log_message(f"Parameters: {vars(args)}")
    deg_data = load_data(args.DEGs, args.h5_key)
    if deg_data.shape[1] != 1:
        raise ValueError("DEG file must have exactly one label column and gene IDs as index")
    deg_data.columns = ["DE_Label"]
    label_text = deg_data["DE_Label"].astype("string")
    if deg_data["DE_Label"].isna().any() or label_text.str.strip().eq("").any():
        raise ValueError("DEG labels must not contain null or blank values")
    encoder = LabelEncoder()
    deg_data["DE_Label_Encoded"] = encoder.fit_transform(deg_data["DE_Label"])
    if len(encoder.classes_) != 2:
        raise ValueError("Binary classification requires exactly two DEG classes")
    log_message(f"Classes: {encoder.classes_.tolist()} -> [0, 1]. SHAP sign only pushes toward/away from encoded class 1.")

    all_results = {}; all_legacy_folds = []
    for i, feature_path in enumerate(args.TF_features, 1):
        log_message(f"Processing feature {i}/{len(args.TF_features)}: {feature_path}")
        result = process_feature(args, feature_path, deg_data, encoder)
        if result is None or result["legacy_result"] is None:
            continue
        all_results[result["feature_name"]] = result["legacy_result"]
        all_legacy_folds.append(result["legacy_folds"])
    if all_legacy_folds:
        metrics = pd.concat(all_legacy_folds, ignore_index=True)
        metrics.drop(columns=["fpr", "tpr"]).to_csv(f"{args.out_prefix}_all_performance_metrics.csv", index=False)
    colors = get_custom_colors(len(args.TF_features))
    if len(all_results) == 1:
        name = next(iter(all_results))
        plot_single_feature_performance(all_results[name], args.out_prefix, name, colors)
    elif len(all_results) > 1:
        plot_performance_comparison(all_results, args.out_prefix, colors)
    if all_results:
        plot_roc_curves(all_results, args.out_prefix, colors)
    log_message("Script finished successfully.")


if __name__ == "__main__":
    main()
