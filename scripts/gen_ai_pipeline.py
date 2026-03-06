#!/usr/bin/env python3
"""
microbiome_genai_explainer_v2.py
--------------------------------
Enhanced GenAI-based explanation of microbiome ML results.

Generates 3 interpretive reports:
auto_genai_summary.txt       – plain-language model performance
auto_genai_taxa_insights.txt – biological interpretation of top taxa
auto_genai_hypotheses.txt    – new mechanistic hypotheses for study

Usage:
  python scripts/microbiome_genai_explainer_v2.py \
      --results results/run_2025-10-12_16-28-41_v14_3_hybrid_cv \
      --hf_token hf_xxxxxxxxxxxxxxxxxxxxxx \
      --model mistralai/Mixtral-8x7B-Instruct-v0.1
"""

import os, argparse, pandas as pd
from huggingface_hub import InferenceClient

# ---------------- Helper: call HF model ----------------
def ask_genai(prompt, token, model="mistralai/Mixtral-8x7B-Instruct-v0.1"):
    """Send prompt to Hugging Face text-generation/chat model."""
    try:
        client = InferenceClient(token=token)
        # Try standard text generation first
        response = client.text_generation(prompt, model=model, max_new_tokens=800, temperature=0.6)
        return response
    except Exception:
        print("Switching to chat mode automatically...")
        try:
            messages = [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(model=model, messages=messages, max_tokens=800)
            return response.choices[0].message["content"]
        except Exception as e:
            raise RuntimeError(f"GenAI request failed: {e}")

# ---------------- Main script ----------------
def main(results_dir, hf_token, model):
    print(f"Reading results from: {results_dir}")
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Load metrics if available
    metrics_path = os.path.join(results_dir, "metrics_clr.csv")
    if not os.path.exists(metrics_path):
        metrics_path = os.path.join(results_dir, "final_metrics_v14_3.csv")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError("No metrics CSV found in results directory.")

    metrics = pd.read_csv(metrics_path)
    print(f"Loaded metrics: {metrics.shape[0]} rows, {metrics.shape[1]} cols")

    # Load selected features if available
    features_path = os.path.join(results_dir, "final_selected_features_v14_3.txt")
    selected_features = []
    if os.path.exists(features_path):
        with open(features_path, "r", encoding="utf-8") as f:
            selected_features = [line.strip() for line in f if line.strip()]
    else:
        print("No selected_features file found — skipping taxa section.")

    # Build shared context
    context = (
        "You are an expert bioinformatics assistant. Analyze microbiome ML results "
        "to produce clear, biologically informed summaries suitable for a report."
    )
    metrics_str = metrics.to_string(index=False)

    # ---------- PROMPTS ----------
    prompt_summary = f"""{context}

Here are the model metrics:
{metrics_str}

Write a detailed plain-language summary of model performance, mentioning accuracy, precision, recall, F1, and AUC.
Explain what the numbers imply about how well the model generalizes.
"""
    prompt_taxa = f"""{context}

These are the top features or taxa selected by the model:
{', '.join(selected_features[:25])}

Explain the biological relevance or known associations of these taxa or metadata variables based on microbiome literature.
Focus on potential gut-brain or metabolic connections.
"""
    prompt_hypotheses = f"""{context}

Given the following taxa and metadata variables:
{', '.join(selected_features[:25])}

Propose 3-5 mechanistic hypotheses that could link these microbes or variables to the target phenotype (depression, bipolar, schizophrenia).
Each hypothesis should be phrased in biological terms and testable.
"""

    # ---------- GENERATION ----------
    print("Generating GenAI summary via Hugging Face model...")
    summary_text = ask_genai(prompt_summary, hf_token, model=model)
    taxa_text = ask_genai(prompt_taxa, hf_token, model=model)
    hypotheses_text = ask_genai(prompt_hypotheses, hf_token, model=model)

    # ---------- SAVE ----------
    with open(os.path.join(results_dir, "auto_genai_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text.strip())
    with open(os.path.join(results_dir, "auto_genai_taxa_insights.txt"), "w", encoding="utf-8") as f:
        f.write(taxa_text.strip())
    with open(os.path.join(results_dir, "auto_genai_hypotheses.txt"), "w", encoding="utf-8") as f:
        f.write(hypotheses_text.strip())

    print("\nSaved GenAI outputs:")
    print(f"  - Summary        → {os.path.join(results_dir, 'auto_genai_summary.txt')}")
    print(f"  - Taxa Insights  → {os.path.join(results_dir, 'auto_genai_taxa_insights.txt')}")
    print(f"  - Hypotheses     → {os.path.join(results_dir, 'auto_genai_hypotheses.txt')}")

# ---------------- CLI ----------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="Path to model results directory")
    p.add_argument("--hf_token", required=True, help="Your Hugging Face access token (starts with hf_)")
    p.add_argument("--model", default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="HF model to use (supports chat/text)")
    args = p.parse_args()

    main(args.results, args.hf_token, args.model)
