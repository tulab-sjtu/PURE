#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PURE_CatBoost_SHAP: CatBoost-based Predictive Modeling and SHAP Explanation.

This module serves as the core predictive engine of the PURE framework. It trains
CatBoost classifiers using transcription factor (TF) binding data or gene expression
profiles to predict Differentially Expressed Genes (DEGs). Beyond prediction,
it leverages SHAP (SHapley Additive exPlanations) to quantify feature importance,
providing interpretable insights into regulatory mechanisms.

Key Features:
1.  Robust Data Handling: Supports both CSV and HDF5 (.h5) formats.
2.  Predictive Modeling: Utilizes CatBoost's gradient boosting.
3.  SHAP Analysis: Computes both raw and filtered SHAP values.
4.  Visualization Suite: Generates ROC curves, performance bar charts, and SHAP summary plots.

Author: CS Li
Date: 2026-01-31
Version: 0.0.1
"""

import argparse
import pandas as pd
import numpy as np
import catboost as cb
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score
)
import matplotlib.pyplot as plt
import os
import time

# --- Color Palette & Plotting Style ---

def get_custom_colors(num_colors):
    """Returns a curated list of hex color codes for publication-quality figures."""
    elegant_palette = [
        '#33658A', '#86BBD8', '#2F4858', '#F6AE2D',
        '#9BC53D', '#55dde0', '#F26419', '#758E4F'
    ]
    return [elegant_palette[i % len(elegant_palette)] for i in range(num_colors)]

def set_plot_style():
    """Configures Matplotlib settings for clean, academic-standard aesthetics."""
    plt.style.use('seaborn-v0_8-ticks')
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['legend.frameon'] = False

def log_message(message):
    """Standardized logging utility."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def load_data(filepath, h5_key):
    """
    Robust data loader handling CSV and HDF5 formats with error management.
    """
    log_message(f"Loading data from: {filepath}")
    _, extension = os.path.splitext(filepath)
    try:
        if extension.lower() == '.csv':
            df = pd.read_csv(filepath, index_col=0)
        elif extension.lower() in ['.h5', '.hdf5']:
            try:
                df = pd.read_hdf(filepath, key=h5_key)
            except KeyError:
                with pd.HDFStore(filepath, 'r') as store:
                    available_keys = store.keys()
                raise KeyError(
                    f"Key '{h5_key}' not found in HDF5 file: {filepath}. "
                    f"Available keys are: {available_keys}"
                )
        else:
            raise ValueError(f"Unsupported file format: {extension}. Please use .csv or .h5")
        log_message(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        log_message(f"ERROR: Failed to load data from {filepath}. Reason: {e}")
        raise

# --- Plotting Functions with Dynamic Sizing ---

def plot_single_feature_performance(results, prefix, feature_name, colors):
    """Generates a bar chart of performance metrics for a single feature set."""
    metrics = list(results['mean_scores'].keys())
    means = list(results['mean_scores'].values())
    stds = list(results['std_scores'].values())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(metrics, means, yerr=stds, align='center', alpha=0.9, ecolor='black', capsize=10, color=colors[0])
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Model Performance Metrics: {feature_name}', fontsize=14, pad=20)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    
    output_path = f"{prefix}_{feature_name}_Performance_Metrics.pdf"
    log_message(f"Saving single feature performance plot to: {output_path}")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

def plot_performance_comparison(all_results, prefix, colors):
    """Generates a grouped bar chart comparing multiple feature sets."""
    feature_names = list(all_results.keys())
    num_features = len(feature_names)
    metrics = list(next(iter(all_results.values()))['mean_scores'].keys())
    
    # Dynamically calculate figure width
    dynamic_width = 6 + 2.5 * num_features
    
    x = np.arange(len(metrics))
    width = 0.8 / num_features
    
    fig, ax = plt.subplots(figsize=(dynamic_width, 7))
    
    for i, name in enumerate(feature_names):
        means = list(all_results[name]['mean_scores'].values())
        stds = list(all_results[name]['std_scores'].values())
        offset = width * (i - (num_features - 1) / 2)
        ax.bar(x + offset, means, width, yerr=stds, label=name, color=colors[i], capsize=4)

    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(title='Feature Sets')
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    
    output_path = f"{prefix}_Performance_Comparison.pdf"
    log_message(f"Saving performance comparison plot to: {output_path}")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

def plot_roc_curves(all_results, prefix, colors):
    """Calculates mean ROC curves, saves coordinates, and generates plots."""
    plt.figure(figsize=(6, 6))
    
    for i, (name, results) in enumerate(all_results.items()):
        # Calculate mean ROC curve from all folds
        mean_fpr = np.linspace(0, 1, 100)
        tprs = [np.interp(mean_fpr, fold['fpr'], fold['tpr']) for fold in results['fold_data']]
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
        mean_auc = np.mean([fold['AUC'] for fold in results['fold_data']])
        std_auc = np.std([fold['AUC'] for fold in results['fold_data']])
        
        # --- Save ROC curve data to CSV ---
        roc_data_df = pd.DataFrame({
            'False_Positive_Rate': mean_fpr,
            'True_Positive_Rate': mean_tpr
        })
        roc_data_output_path = f"{prefix}_{name}_ROC_curve_data.csv"
        log_message(f"Saving ROC curve data for '{name}' to: {roc_data_output_path}")
        roc_data_df.to_csv(roc_data_output_path, index=False)
        # ----------------------------------
        
        # Plot the mean ROC curve
        plt.plot(mean_fpr, mean_tpr, color=colors[i], lw=2.5,
                 label=f'{name} (AUC = {mean_auc:.3f} $\\pm$ {std_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, pad=20)
    plt.legend(loc="lower right")
    
    output_path = f"{prefix}_ROC_Curves.pdf"
    log_message(f"Saving ROC curves plot to: {output_path}")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train a CatBoost model to predict DEGs and explain with SHAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Input/Output Arguments ---
    parser.add_argument('--out_prefix', type=str, required=True, help="Prefix for all output files.")
    parser.add_argument('--TF_features', type=str, nargs='+', required=True, help="One or more paths to feature files (CSV or H5).")
    parser.add_argument('--DEGs', type=str, required=True, help="Path to the DEG file with gene IDs and labels.")
    parser.add_argument('--h5_key', type=str, default='/regulons', help="The key for the DataFrame in HDF5 (h5) files.")

    # --- Model Training Arguments ---
    parser.add_argument('--threads', type=int, default=8, help="Number of threads for CatBoost training.")
    parser.add_argument('--iterations', type=int, default=1000, help="Number of boosting iterations (trees).")
    parser.add_argument('--learning_rate', type=float, default=0.05, help="Learning rate for CatBoost.")
    parser.add_argument('--depth', type=int, default=6, help="Depth of the trees in CatBoost.")
    parser.add_argument('--l2_leaf_reg', type=float, default=3.0, help="L2 regularization coefficient.")
    parser.add_argument('--auto_class_weights', type=str, default='Balanced', choices=['None', 'Balanced'], help="Method to handle class imbalance.")
    # --- Evaluation Arguments ---
    parser.add_argument('--n_splits', type=int, default=5, help="Number of splits for Stratified K-Fold cross-validation.")

    args = parser.parse_args()
    set_plot_style()
    log_message("Script started.")
    log_message(f"Parameters: {vars(args)}")

    # Load Label Data (DEGs)
    deg_data = load_data(args.DEGs, args.h5_key)
    if deg_data.shape[1] != 1:
        raise ValueError("DEG file should have exactly one column for labels, with Gene IDs as the index.")
    deg_data.columns = ['DE_Label']
    
    label_encoder = LabelEncoder()
    deg_data['DE_Label_Encoded'] = label_encoder.fit_transform(deg_data['DE_Label'])
    
    if len(label_encoder.classes_) != 2:
        raise ValueError("This script currently supports binary classification only.")
    label_for_class_0 = label_encoder.classes_[0]
    label_for_class_1 = label_encoder.classes_[1]
    log_message(f"DEG Classes Encoded: {list(label_encoder.classes_)} -> {label_encoder.transform(label_encoder.classes_)}")
    log_message(f"SHAP Interpretation Note: Positive SHAP values predict '{label_for_class_1}', Negative values predict '{label_for_class_0}'.")

    all_model_results = {}
    all_metrics_df = pd.DataFrame()
    colors = get_custom_colors(len(args.TF_features))

    # Iterate over provided feature sets
    for i, feature_path in enumerate(args.TF_features):
        feature_name = os.path.splitext(os.path.basename(feature_path))[0]
        log_message(f"--- Processing Feature Set {i+1}/{len(args.TF_features)}: {feature_name} ---")
        
        feature_data = load_data(feature_path, args.h5_key)
        merged_data = feature_data.join(deg_data, how='inner')
        
        if merged_data.empty:
            log_message(f"Warning: No common genes found between {feature_name} and DEG file. Skipping.")
            continue
        
        log_message(f"Found {len(merged_data)} common genes for this feature set.")

        X = merged_data.drop(columns=['DE_Label', 'DE_Label_Encoded'])
        y = merged_data['DE_Label_Encoded']
        
        log_message("Preprocessing: Applying Z-score standardization to features.")
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
        
        cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        fold_data = []
        log_message(f"Starting {args.n_splits}-fold cross-validation...")
        
        # Cross-validation Loop
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
            log_message(f"  Fold {fold+1}/{args.n_splits}...")
            X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = cb.CatBoostClassifier(iterations=args.iterations, learning_rate=args.learning_rate, depth=args.depth, l2_leaf_reg=args.l2_leaf_reg, auto_class_weights=args.auto_class_weights, thread_count=args.threads, verbose=0, random_seed=42+fold)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred_class = model.predict(X_val)
            fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
            fold_metrics = {'Feature_Set': feature_name, 'Fold': fold + 1, 'Accuracy': accuracy_score(y_val, y_pred_class), 'AUC': auc(fpr, tpr), 'Precision': precision_score(y_val, y_pred_class, zero_division=0), 'Recall': recall_score(y_val, y_pred_class), 'F1_Score': f1_score(y_val, y_pred_class), 'Balanced_Accuracy': balanced_accuracy_score(y_val, y_pred_class), 'fpr': fpr, 'tpr': tpr}
            fold_data.append(fold_metrics)
        
        fold_df = pd.DataFrame(fold_data)
        all_metrics_df = pd.concat([all_metrics_df, fold_df], ignore_index=True)
        mean_scores = fold_df.drop(columns=['Feature_Set', 'Fold', 'fpr', 'tpr']).mean().to_dict()
        std_scores = fold_df.drop(columns=['Feature_Set', 'Fold', 'fpr', 'tpr']).std().to_dict()
        all_model_results[feature_name] = {'fold_data': fold_data, 'mean_scores': mean_scores, 'std_scores': std_scores}
        log_message(f"Cross-validation for {feature_name} complete. Mean AUC: {mean_scores['AUC']:.3f}")

        # Final Model Training for SHAP Analysis
        log_message("Training final model on full dataset for SHAP analysis...")
        final_model = cb.CatBoostClassifier(iterations=args.iterations, learning_rate=args.learning_rate, depth=args.depth, l2_leaf_reg=args.l2_leaf_reg, auto_class_weights=args.auto_class_weights, thread_count=args.threads, verbose=False, random_seed=42)
        final_model.fit(X_scaled, y)
        
        log_message("Calculating SHAP values...")
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_scaled)
        
        shap_df = pd.DataFrame(shap_values, index=X_scaled.index, columns=X_scaled.columns)
        expected_value = explainer.expected_value
        
        # Save Raw SHAP Values
        shap_filename_raw = f"{args.out_prefix}_{feature_name}_SHAP_raw_values_exp_{expected_value:.4f}_pos_is_{label_for_class_1}_neg_is_{label_for_class_0}.csv"
        log_message(f"Saving RAW SHAP contribution matrix to: {shap_filename_raw}")
        shap_df.to_csv(shap_filename_raw)

        # Filter SHAP Values (Zero out if original feature was 0)
        log_message("Filtering SHAP values based on original features (value is 0 if original feature was 0).")
        shap_filtered_df = shap_df.copy()
        zero_mask = (X == 0)
        shap_filtered_df[zero_mask] = 0
        
        shap_filename_filtered = f"{args.out_prefix}_{feature_name}_SHAP_filtered_values_exp_{expected_value:.4f}_pos_is_{label_for_class_1}_neg_is_{label_for_class_0}.csv"
        log_message(f"Saving FILTERED SHAP contribution matrix to: {shap_filename_filtered}")
        shap_filtered_df.to_csv(shap_filename_filtered)
        
        # Generate SHAP Summary Plot
        plt.figure()
        shap.summary_plot(shap_values, X_scaled, show=False, max_display=20, plot_type="dot")
        title_str = (f"SHAP Feature Importance for {feature_name}\n" f"(Positive SHAP predicts '{label_for_class_1}' | Negative SHAP predicts '{label_for_class_0}')")
        plt.title(title_str, fontsize=14, pad=20)
        shap_plot_filename = f"{args.out_prefix}_{feature_name}_SHAP_summary_top20.pdf"
        log_message(f"Saving SHAP summary plot (Top 20) to: {shap_plot_filename}")
        plt.savefig(shap_plot_filename, format='pdf', bbox_inches='tight')
        plt.close()

    if not all_metrics_df.empty:
        metrics_csv_path = f"{args.out_prefix}_all_performance_metrics.csv"
        log_message(f"Saving all performance metrics to: {metrics_csv_path}")
        all_metrics_df.drop(columns=['fpr', 'tpr']).to_csv(metrics_csv_path, index=False)
    
    log_message("--- Generating Final Plots ---")
    if len(all_model_results) > 1:
        plot_performance_comparison(all_model_results, args.out_prefix, colors)
    elif len(all_model_results) == 1:
        feature_name = list(all_model_results.keys())[0]
        plot_single_feature_performance(all_model_results[feature_name], args.out_prefix, feature_name, colors)
    
    if all_model_results:
        plot_roc_curves(all_model_results, args.out_prefix, colors)
    else:
        log_message("No models were trained, skipping plot generation.")
    
    log_message("Script finished successfully.")

if __name__ == '__main__':
    main()