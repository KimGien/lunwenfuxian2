"""
Pairwise Statistical Significance Tests for TRI-DEP Predictions
================================================================

This script performs pairwise statistical significance tests between all model
configurations used in the TRI-DEP multimodal depression detection study.

Two complementary approaches are used:

1. **McNemar's exact test** (per-fold, then combined):
   Compares paired binary predictions (correct/incorrect) for two models on the
   same subjects within each fold. Uses the exact binomial version since fold
   sizes are small (7-8 subjects). Per-fold p-values are combined across folds
   using Fisher's method.

2. **Wilcoxon signed-rank test** on per-fold F1 scores:
   Compares the 5 fold-level F1 scores between two models. This directly tests
   whether one model consistently outperforms the other across folds.

All p-values are corrected for multiple comparisons using the
Benjamini-Hochberg (FDR) procedure.

Usage:
    python pairwise_significance_tests.py
"""

import ast
import itertools
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

PREDICTIONS_DIR = Path(__file__).parent / "predictions"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Friendly display names for the CSV files
MODEL_NAMES = {
    # Unimodal — EEG
    "eeg_cbramod_mumtaz_conv": "EEG (CBraMod Mumtaz + Conv)",
    "eeg_labram_gru_attn": "EEG (LaBraM + GRU+Attn)",
    "eeg_xhand_cnn_lstm": "EEG (Handcrafted + CNN+LSTM)",
    # Unimodal — Speech
    "speech_hubert_bigru_conv": "Speech (HuBERT + BiGRU+Conv)",
    "speech_xlsr_cnn_conv": "Speech (XLSR + CNN+Conv)",
    "speech_mfcc_cnn_lstm": "Speech (MFCC + CNN+LSTM)",
    "speech_prosody_mfcc_bigru_attn": "Speech (Prosody+MFCC + BiGRU+Attn)",
    # Unimodal — Text
    "text_macbert_lstm": "Text (MacBERT + LSTM)",
    "text_bert_cnn": "Text (BERT + CNN)",
    "text_xlnet_lstm": "Text (XLNet + LSTM)",
    "text_mpnet_cnn": "Text (MPNet + CNN)",
    # Fusion — Weighted Averaging
    "wa_eeg_speech": "WA: EEG+Speech",
    "wa_eeg_text": "WA: EEG+Text",
    "wa_speech_text": "WA: Speech+Text",
    "wa_eeg_speech_text": "WA: EEG+Speech+Text",
    # Fusion — Bayesian
    "bf_eeg_speech": "BF: EEG+Speech",
    "bf_eeg_text": "BF: EEG+Text",
    "bf_speech_text": "BF: Speech+Text",
    "bf_eeg_speech_text": "BF: EEG+Speech+Text",
    # Fusion — Majority Voting
    "mv_eeg_speech": "MV: EEG+Speech",
    "mv_eeg_text": "MV: EEG+Text",
    "mv_speech_text": "MV: Speech+Text",
    "mv_eeg_speech_text": "MV: EEG+Speech+Text",
}

# Skip stale/duplicate/legacy files
SKIP_FILES = {
    "text",
    "eeg",
    "speech",  # old aggregate files
    "bayesian_eeg_speech",
    "bayesian_eeg_speech_text",  # old naming
    "bayesian_eeg_text",
    "bayesian_speech_text",
    "mean_eeg_speech",
    "mean_eeg_speech_text",
    "mean_eeg_text",
    "mean_speech_text",
    "speech_xlsr_lstm",
    "speech_hubert_lstm",  # old naming (no encoder suffix)
    "speech_mfcc_lstm",
    "speech_prosody_mfcc_lstm",
}


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────


def load_predictions(predictions_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all prediction CSV files into a dict keyed by model name."""
    models = {}
    for csv_path in sorted(predictions_dir.glob("*.csv")):
        key = csv_path.stem
        if key in SKIP_FILES:
            continue
        df = pd.read_csv(csv_path)
        # Ensure consistent column types
        df["subject_id"] = df["subject_id"].astype(str)
        df["fold"] = df["fold"].astype(int)
        df["ground_truth"] = df["ground_truth"].astype(int)
        df["predicted_label"] = df["predicted_label"].astype(int)
        # Add correctness column
        df["correct"] = (df["ground_truth"] == df["predicted_label"]).astype(int)
        models[key] = df
    return models


# ──────────────────────────────────────────────────────────────────────────────
# Per-fold metrics
# ──────────────────────────────────────────────────────────────────────────────


def compute_fold_metrics(models: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute F1, accuracy, precision, recall per fold for each model."""
    records = []
    for model_name, df in models.items():
        for fold in sorted(df["fold"].unique()):
            fold_df = df[df["fold"] == fold]
            y_true = fold_df["ground_truth"].values
            y_pred = fold_df["predicted_label"].values
            f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            f1_binary = f1_score(y_true, y_pred, average="binary", zero_division=0)
            acc = np.mean(y_true == y_pred)
            records.append(
                {
                    "model": model_name,
                    "display_name": MODEL_NAMES.get(model_name, model_name),
                    "fold": fold,
                    "f1": f1_macro,
                    "f1_macro": f1_macro,
                    "f1_weighted": f1_weighted,
                    "f1_binary": f1_binary,
                    "accuracy": acc,
                    "n_subjects": len(fold_df),
                }
            )
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# McNemar's exact test (per fold + Fisher's combination)
# ──────────────────────────────────────────────────────────────────────────────


def mcnemar_exact_test(correct_a: np.ndarray, correct_b: np.ndarray):
    """
    McNemar's exact test comparing two models' binary correctness vectors.

    Returns the two-sided p-value from the exact binomial test.
    b = # subjects where A is wrong but B is right
    c = # subjects where A is right but B is wrong
    Under H0: b ~ Binomial(b+c, 0.5)
    """
    b = np.sum((correct_a == 0) & (correct_b == 1))  # A wrong, B right
    c = np.sum((correct_a == 1) & (correct_b == 0))  # A right, B wrong
    n = b + c
    if n == 0:
        return 1.0, b, c  # no discordant pairs → no evidence of difference
    # Two-sided exact binomial test
    p_value = stats.binomtest(b, n, 0.5).pvalue
    return p_value, int(b), int(c)


def fisher_combine_pvalues(pvalues: list[float]):
    """
    Fisher's method to combine independent p-values.
    Returns the combined test statistic and p-value.
    Handles p=1.0 and p=0.0 edge cases.
    """
    pvalues = np.array(pvalues, dtype=float)
    # Clamp to avoid log(0)
    pvalues = np.clip(pvalues, 1e-300, 1.0)
    chi2_stat = -2 * np.sum(np.log(pvalues))
    combined_p = stats.chi2.sf(chi2_stat, df=2 * len(pvalues))
    return chi2_stat, combined_p


# ──────────────────────────────────────────────────────────────────────────────
# Pairwise tests
# ──────────────────────────────────────────────────────────────────────────────


def run_pairwise_tests(models: dict[str, pd.DataFrame], fold_metrics: pd.DataFrame):
    """
    Run pairwise McNemar's + Wilcoxon signed-rank tests between all model pairs.
    """
    model_names = sorted(models.keys())
    folds = sorted(models[model_names[0]]["fold"].unique())
    results = []

    for model_a, model_b in itertools.combinations(model_names, 2):
        df_a = models[model_a].set_index(["subject_id", "fold"])
        df_b = models[model_b].set_index(["subject_id", "fold"])

        # Ensure aligned subjects
        common_idx = df_a.index.intersection(df_b.index)
        df_a = df_a.loc[common_idx].sort_index()
        df_b = df_b.loc[common_idx].sort_index()

        # ── McNemar's per fold, then Fisher's combination ──
        fold_pvalues = []
        total_b, total_c = 0, 0
        for fold in folds:
            mask = df_a.index.get_level_values("fold") == fold
            corr_a = df_a.loc[mask, "correct"].values
            corr_b = df_b.loc[mask, "correct"].values
            p_fold, b, c = mcnemar_exact_test(corr_a, corr_b)
            fold_pvalues.append(p_fold)
            total_b += b
            total_c += c

        _, mcnemar_combined_p = fisher_combine_pvalues(fold_pvalues)

        # Also compute global McNemar across all subjects pooled
        corr_a_all = df_a["correct"].values
        corr_b_all = df_b["correct"].values
        mcnemar_global_p, _, _ = mcnemar_exact_test(corr_a_all, corr_b_all)

        # ── Wilcoxon signed-rank test on per-fold F1 scores ──
        f1_a = (
            fold_metrics[fold_metrics["model"] == model_a]
            .sort_values("fold")["f1"]
            .values
        )
        f1_b = (
            fold_metrics[fold_metrics["model"] == model_b]
            .sort_values("fold")["f1"]
            .values
        )

        f1_diff = f1_a - f1_b
        if np.all(f1_diff == 0):
            wilcoxon_p = 1.0
        else:
            try:
                _, wilcoxon_p = stats.wilcoxon(f1_a, f1_b, alternative="two-sided")
            except ValueError:
                # All differences are zero or only one non-zero
                wilcoxon_p = 1.0

        mean_f1_a = np.mean(f1_a)
        mean_f1_b = np.mean(f1_b)
        std_f1_a = np.std(f1_a, ddof=1)
        std_f1_b = np.std(f1_b, ddof=1)

        results.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "display_a": MODEL_NAMES.get(model_a, model_a),
                "display_b": MODEL_NAMES.get(model_b, model_b),
                "mean_f1_a": mean_f1_a,
                "std_f1_a": std_f1_a,
                "mean_f1_b": mean_f1_b,
                "std_f1_b": std_f1_b,
                "f1_diff": mean_f1_a - mean_f1_b,
                "discordant_b": total_b,  # A wrong, B right
                "discordant_c": total_c,  # A right, B wrong
                "mcnemar_combined_p": mcnemar_combined_p,
                "mcnemar_global_p": mcnemar_global_p,
                "wilcoxon_p": wilcoxon_p,
                "fold_f1_a": f1_a.tolist(),
                "fold_f1_b": f1_b.tolist(),
            }
        )

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# Multiple testing correction (Benjamini-Hochberg FDR)
# ──────────────────────────────────────────────────────────────────────────────


def apply_fdr_correction(results_df: pd.DataFrame) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction to all p-value columns."""
    for col in ["mcnemar_combined_p", "mcnemar_global_p", "wilcoxon_p"]:
        pvals = results_df[col].values
        n = len(pvals)
        sorted_idx = np.argsort(pvals)
        sorted_pvals = pvals[sorted_idx]
        # BH correction
        adjusted = np.zeros(n)
        for i in range(n - 1, -1, -1):
            rank = i + 1
            if i == n - 1:
                adjusted[i] = sorted_pvals[i]
            else:
                adjusted[i] = min(adjusted[i + 1], sorted_pvals[i] * n / rank)
        adjusted = np.minimum(adjusted, 1.0)
        # Map back to original order
        result = np.zeros(n)
        result[sorted_idx] = adjusted
        results_df[f"{col}_fdr"] = result

    return results_df


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────


def format_p(p: float) -> str:
    """Format p-value for display."""
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def significance_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def generate_reports(
    results_df: pd.DataFrame, fold_metrics: pd.DataFrame, output_dir: Path
):
    """Generate all output files."""

    # ── 1. Full pairwise results CSV ──
    out_cols = [
        "display_a",
        "display_b",
        "mean_f1_a",
        "std_f1_a",
        "mean_f1_b",
        "std_f1_b",
        "f1_diff",
        "discordant_b",
        "discordant_c",
        "mcnemar_global_p",
        "mcnemar_global_p_fdr",
        "mcnemar_combined_p",
        "mcnemar_combined_p_fdr",
        "wilcoxon_p",
        "wilcoxon_p_fdr",
    ]
    results_df[out_cols].to_csv(output_dir / "pairwise_all_results.csv", index=False)

    # ── 2. Per-fold F1 scores CSV ──
    fold_metrics.to_csv(output_dir / "per_fold_metrics.csv", index=False)

    # ── 3. Key comparisons summary (the ones the reviewer cares about) ──
    key_pairs = [
        # Trimodal vs best unimodal (text)
        ("wa_eeg_speech_text", "text_macbert_lstm"),
        ("bf_eeg_speech_text", "text_macbert_lstm"),
        ("mv_eeg_speech_text", "text_macbert_lstm"),
        # Trimodal vs best unimodal (speech)
        ("wa_eeg_speech_text", "speech_hubert_bigru_conv"),
        ("bf_eeg_speech_text", "speech_hubert_bigru_conv"),
        # Best bimodal vs best unimodal
        ("wa_speech_text", "text_macbert_lstm"),
        ("wa_speech_text", "speech_hubert_bigru_conv"),
        # Trimodal vs best bimodal
        ("wa_eeg_speech_text", "wa_speech_text"),
        ("bf_eeg_speech_text", "wa_speech_text"),
        # Cross-modality unimodal comparisons
        ("text_macbert_lstm", "speech_hubert_bigru_conv"),
        ("text_macbert_lstm", "eeg_cbramod_mumtaz_conv"),
        ("speech_hubert_bigru_conv", "eeg_cbramod_mumtaz_conv"),
    ]

    # ── 4. Readable text report ──
    lines = []
    lines.append("=" * 90)
    lines.append("PAIRWISE STATISTICAL SIGNIFICANCE TESTS — TRI-DEP")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Tests performed:")
    lines.append("  1. McNemar's exact test (pooled across all subjects)")
    lines.append("  2. McNemar's exact test per fold + Fisher's combination")
    lines.append("  3. Wilcoxon signed-rank test on per-fold F1 scores")
    lines.append(
        "  All p-values also reported after Benjamini-Hochberg FDR correction."
    )
    lines.append("")
    lines.append(
        "Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant"
    )
    lines.append("")

    # Section: Per-fold F1 scores for all models
    lines.append("-" * 90)
    lines.append("PER-FOLD F1 SCORES")
    lines.append("-" * 90)
    model_order = [
        "eeg_cbramod_mumtaz_conv",
        "eeg_labram_gru_attn",
        "eeg_xhand_cnn_lstm",
        "speech_hubert_bigru_conv",
        "speech_xlsr_cnn_conv",
        "speech_mfcc_cnn_lstm",
        "speech_prosody_mfcc_bigru_attn",
        "text_macbert_lstm",
        "text_bert_cnn",
        "text_xlnet_lstm",
        "text_mpnet_cnn",
        "wa_eeg_speech",
        "wa_eeg_text",
        "wa_speech_text",
        "wa_eeg_speech_text",
        "bf_eeg_speech",
        "bf_eeg_text",
        "bf_speech_text",
        "bf_eeg_speech_text",
        "mv_eeg_speech",
        "mv_eeg_text",
        "mv_speech_text",
        "mv_eeg_speech_text",
    ]
    for m in model_order:
        mdf = fold_metrics[fold_metrics["model"] == m].sort_values("fold")
        f1s = mdf["f1"].values
        display = MODEL_NAMES.get(m, m)
        fold_str = "  ".join([f"F{i + 1}={f:.3f}" for i, f in enumerate(f1s)])
        lines.append(
            f"  {display:40s}  {fold_str}  | Mean={np.mean(f1s):.3f} ± {np.std(f1s, ddof=1):.3f}"
        )
    lines.append("")

    # Section: Key comparisons
    lines.append("-" * 90)
    lines.append("KEY PAIRWISE COMPARISONS")
    lines.append("-" * 90)
    lines.append("")

    for ma, mb in key_pairs:
        # Find the row (could be in either order)
        row = results_df[
            ((results_df["model_a"] == ma) & (results_df["model_b"] == mb))
            | ((results_df["model_a"] == mb) & (results_df["model_b"] == ma))
        ]
        if row.empty:
            continue
        row = row.iloc[0]

        # Determine display order: always show the first in the pair first
        if row["model_a"] == ma:
            da, db = row["display_a"], row["display_b"]
            f1a, sa = row["mean_f1_a"], row["std_f1_a"]
            f1b, sb = row["mean_f1_b"], row["std_f1_b"]
        else:
            da, db = row["display_b"], row["display_a"]
            f1a, sa = row["mean_f1_b"], row["std_f1_b"]
            f1b, sb = row["mean_f1_a"], row["std_f1_a"]

        lines.append(f"  {da}  vs  {db}")
        lines.append(
            f"    F1: {f1a:.3f}±{sa:.3f}  vs  {f1b:.3f}±{sb:.3f}  (Δ = {f1a - f1b:+.3f})"
        )
        lines.append(
            f"    Discordant pairs: b={row['discordant_b']}, c={row['discordant_c']}"
        )
        lines.append(
            f"    McNemar (pooled):   p = {format_p(row['mcnemar_global_p']):>10s} {significance_marker(row['mcnemar_global_p']):>3s}  "
            f"(FDR: {format_p(row['mcnemar_global_p_fdr'])} {significance_marker(row['mcnemar_global_p_fdr'])})"
        )
        lines.append(
            f"    McNemar (Fisher):   p = {format_p(row['mcnemar_combined_p']):>10s} {significance_marker(row['mcnemar_combined_p']):>3s}  "
            f"(FDR: {format_p(row['mcnemar_combined_p_fdr'])} {significance_marker(row['mcnemar_combined_p_fdr'])})"
        )
        lines.append(
            f"    Wilcoxon signed-rank: p = {format_p(row['wilcoxon_p']):>10s} {significance_marker(row['wilcoxon_p']):>3s}  "
            f"(FDR: {format_p(row['wilcoxon_p_fdr'])} {significance_marker(row['wilcoxon_p_fdr'])})"
        )
        lines.append("")

    # Section: Full pairwise matrix (condensed)
    lines.append("-" * 90)
    lines.append("FULL PAIRWISE COMPARISON TABLE (all 105 pairs)")
    lines.append("-" * 90)
    lines.append("")
    header = f"  {'Model A':40s} {'Model B':40s} {'ΔF1':>7s} {'McN(p)':>10s} {'McN(FDR)':>10s} {'Wilc(p)':>10s} {'Wilc(FDR)':>10s} {'Sig':>4s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for _, row in results_df.sort_values("wilcoxon_p").iterrows():
        sig = significance_marker(
            min(row["mcnemar_global_p_fdr"], row["wilcoxon_p_fdr"])
        )
        lines.append(
            f"  {row['display_a']:40s} {row['display_b']:40s} "
            f"{row['f1_diff']:>+7.3f} "
            f"{format_p(row['mcnemar_global_p']):>10s} "
            f"{format_p(row['mcnemar_global_p_fdr']):>10s} "
            f"{format_p(row['wilcoxon_p']):>10s} "
            f"{format_p(row['wilcoxon_p_fdr']):>10s} "
            f"{sig:>4s}"
        )
    lines.append("")

    # Section: Summary interpretation
    lines.append("-" * 90)
    lines.append("SUMMARY")
    lines.append("-" * 90)
    lines.append("")

    # Count significant differences
    n_sig_mcnemar = (results_df["mcnemar_global_p_fdr"] < 0.05).sum()
    n_sig_wilcoxon = (results_df["wilcoxon_p_fdr"] < 0.05).sum()
    n_total = len(results_df)

    lines.append(f"  Total pairwise comparisons: {n_total}")
    lines.append(f"  Significant by McNemar (FDR < 0.05): {n_sig_mcnemar}")
    lines.append(f"  Significant by Wilcoxon (FDR < 0.05): {n_sig_wilcoxon}")
    lines.append("")
    lines.append(
        "  Note: With N=38 subjects (7-8 per fold) and 5 folds, statistical power"
    )
    lines.append(
        "  is limited. Non-significant results should be interpreted as 'insufficient"
    )
    lines.append("  evidence to reject H0' rather than 'no difference'.")
    lines.append("")

    report = "\n".join(lines)
    with open(output_dir / "significance_report.txt", "w") as f:
        f.write(report)

    return report


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX table for thesis
# ──────────────────────────────────────────────────────────────────────────────


def generate_latex_table(results_df: pd.DataFrame, output_dir: Path):
    """Generate a LaTeX table of key pairwise comparisons for the thesis."""

    key_pairs = [
        # Trimodal vs best unimodal (text)
        ("wa_eeg_speech_text", "text_macbert_lstm"),
        ("bf_eeg_speech_text", "text_macbert_lstm"),
        ("mv_eeg_speech_text", "text_macbert_lstm"),
        # Trimodal vs best unimodal (speech)
        ("wa_eeg_speech_text", "speech_hubert_bigru_conv"),
        ("bf_eeg_speech_text", "speech_hubert_bigru_conv"),
        # Best bimodal vs best unimodal
        ("wa_speech_text", "text_macbert_lstm"),
        ("wa_speech_text", "speech_hubert_bigru_conv"),
        # Trimodal vs best bimodal
        ("wa_eeg_speech_text", "wa_speech_text"),
        ("bf_eeg_speech_text", "wa_speech_text"),
        # Unimodal cross-comparisons
        ("text_macbert_lstm", "speech_hubert_bigru_conv"),
        ("text_macbert_lstm", "eeg_cbramod_mumtaz_conv"),
        ("speech_hubert_bigru_conv", "eeg_cbramod_mumtaz_conv"),
    ]

    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Model A} & \textbf{Model B} & \textbf{$\Delta$F1} & \textbf{McNemar $p$} & \textbf{Wilcoxon $p$} \\"
    )
    lines.append(r"\midrule")

    for ma, mb in key_pairs:
        row = results_df[
            ((results_df["model_a"] == ma) & (results_df["model_b"] == mb))
            | ((results_df["model_a"] == mb) & (results_df["model_b"] == ma))
        ]
        if row.empty:
            continue
        row = row.iloc[0]

        if row["model_a"] == ma:
            da, db = row["display_a"], row["display_b"]
            delta = row["mean_f1_a"] - row["mean_f1_b"]
        else:
            da, db = row["display_b"], row["display_a"]
            delta = row["mean_f1_b"] - row["mean_f1_a"]

        mcn_p = row["mcnemar_global_p_fdr"]
        wil_p = row["wilcoxon_p_fdr"]

        # Escape special chars for LaTeX
        da_tex = da.replace("+", "$+$").replace(":", "\\text{:}")
        db_tex = db.replace("+", "$+$").replace(":", "\\text{:}")

        sig_mcn = significance_marker(mcn_p)
        sig_wil = significance_marker(wil_p)
        mcn_str = f"{format_p(mcn_p)}" + (
            f"$^{{{sig_mcn}}}$" if sig_mcn != "ns" else ""
        )
        wil_str = f"{format_p(wil_p)}" + (
            f"$^{{{sig_wil}}}$" if sig_wil != "ns" else ""
        )

        lines.append(
            f"  {da_tex} & {db_tex} & {delta:+.3f} & {mcn_str} & {wil_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(
        r"\caption{Pairwise statistical significance tests for 12 selected "
        r"comparisons from the 253 total pairs across the 23 configurations "
        r"reported in Tables~\ref{tab:unimodal_results}--"
        r"\ref{tab:multimodal_results} (11 best unimodal feature--model pairs "
        r"$+$ 12 fusion configurations). The selection covers all three trimodal "
        r"fusion methods against the best unimodal models, the best bimodal "
        r"(WA: Speech$+$Text) against the best unimodal models, trimodal against "
        r"the best bimodal, and cross-modality unimodal comparisons. "
        r"$\Delta$F1 is the difference in mean macro-F1. "
        r"McNemar's exact test evaluates whether two classifiers make significantly "
        r"different errors on the same subjects ($N{=}38$); the Wilcoxon signed-rank "
        r"test assesses whether per-fold macro-F1 differences are consistent across "
        r"folds ($n{=}5$). With $n{=}5$, the minimum achievable Wilcoxon $p$-value "
        r"is $0.0625$. No comparison reaches significance at $\alpha{=}0.05$. "
        r"``Best'' refers to the best-performing unimodal configuration per modality "
        r"(bold rows in Table~\ref{tab:unimodal_results}): "
        r"Best Text\,=\,$\mathbf{X}_{\text{MacBERT}}$+LSTM, "
        r"Best Speech\,=\,$\mathbf{X}_{\text{HuBERT}}$+BiGRU+CNN+MaxPool, "
        r"Best EEG\,=\,$\mathbf{X}_{\text{CBraMod Mumtaz}}$+CNN. "
        r"WA\,=\,Weighted Averaging, BF\,=\,Bayesian Fusion.}"
    )
    lines.append(r"\label{tab:significance_tests}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    with open(output_dir / "significance_table.tex", "w") as f:
        f.write(latex)

    return latex


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    print("Loading predictions...")
    models = load_predictions(PREDICTIONS_DIR)
    print(f"  Loaded {len(models)} model prediction files.")

    print("\nComputing per-fold metrics...")
    fold_metrics = compute_fold_metrics(models)

    print("Running pairwise significance tests (all 105 pairs)...")
    results_df = run_pairwise_tests(models, fold_metrics)

    print("Applying FDR correction...")
    results_df = apply_fdr_correction(results_df)

    print("Generating reports...")
    report = generate_reports(results_df, fold_metrics, OUTPUT_DIR)
    print(report)

    print("\nGenerating LaTeX table...")
    latex = generate_latex_table(results_df, OUTPUT_DIR)
    print(latex)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print(f"  - pairwise_all_results.csv  (full results)")
    print(f"  - per_fold_metrics.csv      (F1 per fold per model)")
    print(f"  - significance_report.txt   (readable report)")
    print(f"  - significance_table.tex    (LaTeX table for thesis)")


if __name__ == "__main__":
    main()
