import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

def load_results(results_dir: str, correct_answer: str) -> Dict[str, pd.DataFrame]:
    """
    Load all model results from the specified directory and compute hit accuracy.
    Args:
        results_dir: Directory containing results CSV files.
        correct_answer: The correct answer string to check for hits.
    Returns:
        Dictionary mapping model names to their results DataFrame.
    """
    result_files = [f for f in os.listdir(results_dir) if f.startswith("needle_haystack_niah_") and f.endswith(".csv")]
    model_results = {}
    for fname in result_files:
        model_name = fname.split("needle_haystack_niah_")[1].replace(".csv", "")
        df = pd.read_csv(os.path.join(results_dir, fname))
        df["is_hit"] =  df["answer"].fillna("").str.strip().str.lower().apply(lambda x: correct_answer.lower() in x)
        model_results[model_name] = df
    return model_results

def plot_accuracy_per_model(model_results: Dict[str, pd.DataFrame], out_dir: str) -> None:
    """
    Plot and save a bar chart of overall accuracy per model.
    Args:
        model_results: Dictionary of model results DataFrames.
        out_dir: Output directory for saving the plot.
    """
    os.makedirs(os.path.join(out_dir, "performance"), exist_ok=True)
    plt.figure(figsize=(8, 5))
    accs = [df["is_hit"].mean() for df in model_results.values()]
    plt.bar(model_results.keys(), accs)
    plt.ylabel("Accuracy (Hit Rate)")
    plt.title("Overall Accuracy per Model")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "performance", "accuracy_per_model.png"))
    plt.close()

def plot_accuracy_by_tokens(model_results: Dict[str, pd.DataFrame], token_steps: List[int], out_dir: str) -> None:
    """
    Plot and save accuracy by number of tokens for each model.
    Args:
        model_results: Dictionary of model results DataFrames.
        token_steps: List of token bin edges.
        out_dir: Output directory for saving the plot.
    """
    os.makedirs(os.path.join(out_dir, "performance"), exist_ok=True)
    plt.figure(figsize=(10, 6))
    bin_edges = [0] + token_steps + [float('inf')]
    bin_labels = token_steps + [f">{token_steps[-1]}"]
    for model, df in model_results.items():
        df['token_bin'] = pd.cut(df['n_tokens'], bins=bin_edges, labels=bin_labels, right=False)
        grouped = df.groupby('token_bin', observed=False)["is_hit"].mean().reindex(bin_labels)
        plot_x = [int(str(l).replace('>', '')) if str(l).isdigit() else token_steps[-1]*1.2 for l in bin_labels]
        plt.plot(plot_x, grouped, marker="o", label=model)
    plt.xscale("log")
    plt.xlabel("Number of Tokens (log scale)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Number of Tokens (per Model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "performance", "accuracy_by_tokens.png"))
    plt.close()

def plot_heatmaps(model_results: Dict[str, pd.DataFrame], token_steps: List[int], out_dir: str, all_models: bool=False) -> None:
    """
    Plot and save heatmaps of accuracy by token bin and needle position for each model and all models.
    Args:
        model_results: Dictionary of model results DataFrames.
        token_steps: List of token bin edges.
        out_dir: Output directory for saving the plots.
        all_models: If True, plot combined and average heatmaps for all models.
    """
    os.makedirs(os.path.join(out_dir, "performance"), exist_ok=True)
    bin_edges = [0] + token_steps + [float('inf')]
    bin_labels = token_steps + [f">{token_steps[-1]}"]
    if all_models:
        combined = []
        for model, df in model_results.items():
            df['token_bin'] = pd.cut(df['n_tokens'], bins=bin_edges, labels=bin_labels, right=False)
            df['model'] = model
            combined.append(df)
        all_df = pd.concat(combined)
        plt.figure(figsize=(12, 8))
        for model in all_df['model'].unique():
            heatmap_data = all_df[all_df['model'] == model].groupby(['token_bin', 'needle_position'], observed=False)["is_hit"].mean().unstack(fill_value=0).reindex(bin_labels)
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
            plt.title(f"Accuracy Heatmap (Model: {model})")
            plt.ylabel("Number of Tokens")
            plt.xlabel("Needle Position")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "performance", f"heatmap_{model}.png"))
            plt.close()
        avg_heatmap = all_df.groupby(['token_bin', 'needle_position'], observed=False)["is_hit"].mean().unstack(fill_value=0).reindex(bin_labels)
        plt.figure(figsize=(12, 8))
        sns.heatmap(avg_heatmap, annot=True, fmt=".2f", cmap="viridis")
        plt.title("Accuracy Heatmap (All Models)")
        plt.ylabel("Number of Tokens")
        plt.xlabel("Needle Position")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "performance", "heatmap_all_models.png"))
        plt.close()
    else:
        for model, df in model_results.items():
            df['token_bin'] = pd.cut(df['n_tokens'], bins=bin_edges, labels=bin_labels, right=False)
            heatmap_data = df.groupby(['token_bin', 'needle_position'], observed=False)["is_hit"].mean().unstack(fill_value=0).reindex(bin_labels)
            plt.figure(figsize=(8, 6))
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
            plt.title(f"Accuracy Heatmap (Model: {model})")
            plt.ylabel("Number of Tokens")
            plt.xlabel("Needle Position")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "performance", f"heatmap_{model}.png"))
            plt.close()

def plot_accuracy_by_position(model_results: Dict[str, pd.DataFrame], out_dir: str) -> None:
    """
    Plot and save accuracy by needle position for each model.
    Args:
        model_results: Dictionary of model results DataFrames.
        out_dir: Output directory for saving the plot.
    """
    os.makedirs(os.path.join(out_dir, "performance"), exist_ok=True)
    plt.figure(figsize=(10, 6))
    for model, df in model_results.items():
        grouped = df.groupby("needle_position", observed=False)["is_hit"].mean()
        plt.plot(grouped.index, grouped.values, marker="o", label=model)
    plt.xlabel("Needle Position")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Needle Position (per Model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "performance", "accuracy_by_position.png"))
    plt.close()

def plot_accuracy_by_distractors(model_results: Dict[str, pd.DataFrame], out_dir: str) -> None:
    """
    Plot and save accuracy by number of distractors for each model.
    Args:
        model_results: Dictionary of model results DataFrames.
        out_dir: Output directory for saving the plot.
    """
    os.makedirs(os.path.join(out_dir, "performance"), exist_ok=True)
    plt.figure(figsize=(10, 6))
    for model, df in model_results.items():
        grouped = df.groupby("n_distractors", observed=False)["is_hit"].mean()
        plt.plot(grouped.index, grouped.values, marker="o", label=model)
    plt.xlabel("Number of Distractors")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Number of Distractors (per Model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "performance", "accuracy_by_distractors.png"))
    plt.close()

if __name__ == "__main__":
    RESULTS_DIR = "results"
    REPORT_DIR = "reports"
    CORRECT_ANSWER = "Alhazred’s Codex"
    TOKEN_STEPS = [100, 256, 1024, 2048, 4096, 8192, 10000, 12000]
    if 100 not in TOKEN_STEPS:
        TOKEN_STEPS = [100] + TOKEN_STEPS
    model_results = load_results(RESULTS_DIR, CORRECT_ANSWER)
    plot_accuracy_per_model(model_results, REPORT_DIR)
    plot_accuracy_by_tokens(model_results, TOKEN_STEPS, REPORT_DIR)
    plot_heatmaps(model_results, TOKEN_STEPS, REPORT_DIR, all_models=False)
    plot_heatmaps(model_results, TOKEN_STEPS, REPORT_DIR, all_models=True)
    plot_accuracy_by_position(model_results, REPORT_DIR)
    plot_accuracy_by_distractors(model_results, REPORT_DIR)
