import os
import re
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from kneed import KneeLocator
import matplotlib.pyplot as plt
from typing import List, Tuple

openai = OpenAI()


def load_phrases_and_embeddings() -> Tuple[pd.DataFrame, List[str], np.ndarray]:
    """
    Load phrases and embeddings from CSV, or generate from text files if not present.
    Returns:
        Tuple of (DataFrame, list of phrases, numpy array of embeddings).
    """
    csv_path = 'data/lovecraft_embeddings.txt'
    if os.path.exists(csv_path):
        return _load_phrases_and_embeddings_from_csv(csv_path)
    else:
        return _generate_phrases_and_embeddings_from_txt(csv_path)


def _load_phrases_and_embeddings_from_csv(csv_path: str) -> Tuple[pd.DataFrame, List[str], np.ndarray]:
    """
    Load phrases and their embeddings from a CSV file.
    Args:
        csv_path (str): Path to the CSV file containing phrase and embedding data.
    Returns:
        tuple:
            - df (pd.DataFrame): DataFrame with columns ['phrase', 'origin', 'phrase_number'].
            - phrases (list[str]): List of phrases.
            - origins (list[str]): List of origin labels for each phrase.
            - phrase_numbers (list[int]): List of phrase indices for each phrase.
            - embeddings (np.ndarray): Array of phrase embeddings.
    """
    df = pd.read_csv(csv_path)
    phrases = df['phrase'].tolist()
    origins = df['origin'].tolist()
    phrase_numbers = df['phrase_number'].tolist()
    emb_cols = [col for col in df.columns if col.startswith('emb_')]
    embeddings = np.vstack(df[emb_cols].values)

    df = df[['phrase', 'origin', 'phrase_number']]
    return df, phrases, origins, phrase_numbers, embeddings


def _generate_phrases_and_embeddings_from_txt(csv_path: str) -> Tuple[pd.DataFrame, List[str], np.ndarray]:
    """
    Generate phrases and their embeddings from a list of Lovecraft text files.
    Args:
        txt_files (list[tuple[str, str]]): List of tuples containing file paths and their origin labels.
    Returns:
        tuple:
            - df (pd.DataFrame): DataFrame with columns ['phrase', 'origin', 'phrase_number'].
            - phrases (list[str]): List of extracted phrases.
            - origins (list[str]): List of origin labels for each phrase.
            - phrase_numbers (list[int]): List of phrase indices for each phrase.
            - embeddings (np.ndarray): Array of phrase embeddings.
    """
    txt_files = [
        ('data/the_colour_out_of_space.txt', 'the_colour_out_of_space'),
        ('data/at_the_mountains_of_madness.txt', 'at_the_mountains_of_madness'),
        ('data/call_of_cthulhu.txt', 'call_of_cthulhu'),
    ]
    phrases = []
    origins = []
    phrase_numbers = []
    for txt_path, origin in txt_files:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        file_phrases = chunk_by_phrase(text)
        file_phrases = [p for p in file_phrases if p.strip() and p.strip() not in ['."', '.']]
        phrases.extend(file_phrases)
        origins.extend([origin] * len(file_phrases))
        phrase_numbers.extend(list(range(1, len(file_phrases) + 1)))
    df = pd.DataFrame({'phrase': phrases, 'origin': origins, 'phrase_number': phrase_numbers})
    embeddings = get_embeddings(phrases)
    emb_cols = {f'emb_{i}': embeddings[:, i] for i in range(embeddings.shape[1])}
    df = pd.concat([df, pd.DataFrame(emb_cols)], axis=1)
    df.to_csv(csv_path, index=False)
    df = df[['phrase', 'origin', 'phrase_number']]
    return df, phrases, origins, phrase_numbers, embeddings

def chunk_by_phrase(text: str) -> List[str]:
    """
    Split text into phrases by sentence-ending punctuation, including quotes.
    Args:
        text: Input text string.
    Returns:
        List of phrase strings.
    """
    # Split by sentence-ending punctuation, including quotes, without lookbehind
    # Use raw string and escape quotes properly
    phrases = re.split(r'([.!?]["\'\u201d\u201c]?)(?:\s+)', text)
    # Recombine punctuation with preceding text, then filter
    result = []
    buffer = ''
    for i, part in enumerate(phrases):
        if i % 2 == 0:
            buffer = part
        else:
            buffer += part
            result.append(buffer.strip())
            buffer = ''
    if buffer:
        result.append(buffer.strip())
    return [p for p in result if p and not re.fullmatch(r'[.?!"\'\u201d\u201c]+', p)]

def get_embeddings(phrases: List[str], model: str = "text-embedding-3-large") -> np.ndarray:
    """
    Generate embeddings for a list of phrases using the specified model.
    Args:
        phrases: List of phrases to embed.
        model: Embedding model name.
    Returns:
        Numpy array of embeddings.
    """
    # Call OpenAI API for embeddings
    embeddings = []
    for phrase in tqdm(phrases, desc="Generating embeddings"):
        response = openai.embeddings.create(input=phrase, model=model)
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings)


def cluster_embeddings(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    n_neighbors: int = 30,
    min_dist: float = 0.05,
    n_components: int = 50,
    random_state: int = 42,
    min_cluster_size: int = 10,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Cluster embeddings using UMAP for dimensionality reduction and KMeans for clustering.
    Args:
        embeddings: Embeddings to cluster.
        df: DataFrame of phrases and metadata.
        n_neighbors, min_dist, n_components, random_state: UMAP parameters.
        min_cluster_size, min_samples, cluster_selection_epsilon, rerun_threshold: Clustering parameters.
    Returns:
        Tuple of (DataFrame with cluster labels, cluster label array).
    """
    # UMAP dimensionality reduction (only once)
    effective_n_components = min(n_components, len(embeddings) - 1)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=effective_n_components, random_state=random_state, verbose=False)
    umap_embeds = reducer.fit_transform(embeddings)

    # Use KneeLocator to find optimal number of clusters
    max_clusters = min(20, len(df) // min_cluster_size)
    inertias = []
    cluster_range = range(2, max_clusters + 1)
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(umap_embeds)
        inertias.append(kmeans.inertia_)
    knee = KneeLocator(cluster_range, inertias, curve='convex', direction='decreasing')
    n_clusters = knee.knee if knee.knee else 2

    # Plot knee chart and save to file
    plt.figure(figsize=(8, 5))
    plt.plot(list(cluster_range), inertias, marker='o')
    if knee.knee:
        plt.axvline(x=knee.knee, color='r', linestyle='--', label=f'Knee: {knee.knee}')
        plt.scatter([knee.knee], [inertias[knee.knee-2]], color='red', zorder=5)
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Knee plot for optimal number of clusters')
    plt.legend()
    plt.tight_layout()
    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports/knee_plot.png')
    plt.close()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(umap_embeds)
    df['cluster'] = cluster_labels

    return df, cluster_labels

def mmr(
        doc_embeddings: np.ndarray,
        query_embedding: np.ndarray,
        num_items: int = 20,
        balance_factor: float = 0.5
) -> List[int]:
    """
    Selects a diverse yet relevant set of items based on a query using Maximal Marginal Relevance (MMR).

    Args:
        doc_embeddings: Array of document/phrase embeddings (numerical representations).
        query_embedding: Array representing the query's embedding.
        num_items: Number of items to return (default: 20).
        balance_factor: Controls trade-off between relevance and diversity (0 to 1).
                       Higher values prioritize relevance; lower values prioritize diversity.

    Returns:
        List of indices corresponding to selected documents.
    """
    # Initialize lists
    selected_indices = []  # Tracks chosen items
    available_indices = list(range(len(doc_embeddings)))  # Tracks remaining items

    # Calculate similarity of each document to the query
    query_similarities = cosine_similarity(doc_embeddings, query_embedding.reshape(1, -1)).flatten()

    # Start with the most relevant document
    most_relevant_idx = np.argmax(query_similarities)
    selected_indices.append(most_relevant_idx)
    available_indices.remove(most_relevant_idx)

    # Continue selecting items until we reach num_items or run out of candidates
    while len(selected_indices) < num_items and available_indices:
        best_score = -float('inf')
        best_candidate = None

        # Evaluate each remaining candidate
        for candidate_idx in available_indices:
            # How relevant is this candidate to the query?
            relevance = query_similarities[candidate_idx]

            # How similar is this candidate to already selected items? (Take max similarity)
            diversity = max(cosine_similarity(
                doc_embeddings[candidate_idx].reshape(1, -1),
                doc_embeddings[selected_indices]
            ).flatten())

            # Combine relevance and diversity into a single score
            score = balance_factor * relevance - (1 - balance_factor) * diversity

            # Keep track of the best candidate
            if score > best_score:
                best_score = score
                best_candidate = candidate_idx

        # Add the best candidate to our selection
        selected_indices.append(best_candidate)
        available_indices.remove(best_candidate)

    return selected_indices


def plot_coverage(df: pd.DataFrame) -> None:
    """
    Plot and save KDE coverage for the largest cluster across all origins.
    Args:
        df: DataFrame with cluster assignments and metadata.
    """
    import seaborn as sns
    largest_cluster = df['cluster'].value_counts().idxmax()
    cluster_df = df[df['cluster'] == largest_cluster]
    origins_map = {
        'the_colour_out_of_space': 'The Colour Out of Space',
        'at_the_mountains_of_madness': 'At the Mountains of Madness',
        'call_of_cthulhu': 'Call of Cthulhu'
    }
    unique_origins = list(df['origin'].unique())
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=True, sharey=True)
    for idx, origin in enumerate(unique_origins):
        file_df = df[df['origin'] == origin]
        file_len = len(file_df)
        file_cluster_df = cluster_df[cluster_df['origin'] == origin]
        norm_positions = file_cluster_df['phrase_number'] / file_len
        # Hard stop at 0 and 1
        norm_positions = np.clip(norm_positions, 0, 1)
        if len(norm_positions) > 1:
            sns.kdeplot(norm_positions, ax=axes[idx], bw_adjust=0.5, clip=(0, 1))
        axes[idx].set_ylabel('Density')
        axes[idx].set_title(origins_map.get(origin, origin))
        axes[idx].set_xlim(0, 1)
    axes[-1].set_xlabel('Normalized Phrase Position (0=beginning, 1=end)')
    plt.tight_layout()
    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports/coverage_kde.png')
    plt.close()

def plot_all_cluster_coverage(df: pd.DataFrame) -> None:
    """
    Plot and save KDE coverage for all clusters across all origins.
    Args:
        df: DataFrame with cluster assignments and metadata.
    """
    import seaborn as sns
    origins_map = {
        'the_colour_out_of_space': 'The Colour Out of Space',
        'at_the_mountains_of_madness': 'At the Mountains of Madness',
        'call_of_cthulhu': 'Call of Cthulhu'
    }
    clusters = sorted(df['cluster'].unique())
    out_dir = 'reports/coverage_clusters'
    os.makedirs(out_dir, exist_ok=True)
    for cluster_id in clusters:
        cluster_df = df[df['cluster'] == cluster_id]
        n_instances = len(cluster_df)
        unique_origins = list(df['origin'].unique())
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=True, sharey=True)
        for idx, origin in enumerate(unique_origins):
            file_df = df[df['origin'] == origin]
            file_len = len(file_df)
            file_cluster_df = cluster_df[cluster_df['origin'] == origin]
            norm_positions = file_cluster_df['phrase_number'] / file_len
            norm_positions = np.clip(norm_positions, 0, 1)
            if len(norm_positions) > 1:
                sns.kdeplot(norm_positions, ax=axes[idx], bw_adjust=0.5, clip=(0, 1))
            axes[idx].set_ylabel('Density')
            axes[idx].set_title(origins_map.get(origin, origin))
            axes[idx].set_xlim(0, 1)
        axes[-1].set_xlabel('Normalized Phrase Position (0=beginning, 1=end)')
        plt.suptitle(f'Coverage KDE: Cluster {cluster_id} ({n_instances} instances)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{out_dir}/coverage_kde_cluster_{cluster_id}.png')
        plt.close()

def main() -> None:
    """
    Main script for clustering Lovecraft phrases and visualizing cluster coverage.
    Loads phrases and embeddings, clusters them, plots coverage, and prints representatives.
    """
    df, phrases, embeddings = load_phrases_and_embeddings()
    df, cluster_labels = cluster_embeddings(embeddings, df)
    plot_coverage(df)
    plot_all_cluster_coverage(df)

    # Save final clustering results
    df.to_csv('data/lovecraft_final_clusters.csv', index=False)

    # Find largest clusters
    largest_clusters = df['cluster'].value_counts().head(2).index.tolist()
    for cluster_id in largest_clusters:
        cluster_indices = df[df['cluster'] == cluster_id].index.tolist()
        cluster_embeds = embeddings[cluster_indices]

        # Use cluster centroid as query for MMR
        centroid = np.mean(cluster_embeds, axis=0)
        selected = mmr(cluster_embeds, centroid, num_items=20)
        print(f"Cluster {cluster_id} representative phrases:")
        for idx in selected:
            print(df.iloc[cluster_indices[idx]]['phrase'])
        print()



if __name__ == "__main__":
    main()
