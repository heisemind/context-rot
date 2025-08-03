import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

openai = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-large") -> np.ndarray:
    """
    Get the embedding vector for a given text using the specified model.
    Args:
        text: The input text to embed.
        model: The embedding model name.
    Returns:
        Embedding vector as a numpy array.
    """
    response = openai.embeddings.create(input=text, model=model)
    return np.array(response.data[0].embedding)


def main() -> None:
    """
    Main script to evaluate similarity between questions and phrases using embeddings.
    Loads phrases and embeddings, computes similarity, and prints top matches.
    """
    # Load embeddings and phrases
    emb_df = pd.read_csv("data/lovecraft_embeddings.txt")
    emb_cols = [col for col in emb_df.columns if col.startswith("emb_")]
    embeddings = np.vstack(emb_df[emb_cols].values)
    phrases = emb_df['phrase'].tolist()

    # Questions to evaluate
    questions = [
        "What was the name of the book hidden beneath the gnarled oak in the blighted landscape described in the text?",
        "The book hidden beneath the gnarled oak in the blighted landscape was titled Alhazred’s Codex.",
    ]

    results = {}
    for q in questions:
        q_emb = get_embedding(q)
        sims = cosine_similarity(embeddings, q_emb.reshape(1, -1)).flatten()
        top_idx = np.argsort(sims)[-30:][::-1]
        results[q] = [(phrases[i], sims[i]) for i in top_idx]

    # Print results
    for q, matches in results.items():
        print(f"\nQuestion: {q}\nTop 30 similar phrases:")
        for phrase, score in matches:
            print(f"Score: {score:.4f} | {phrase.replace("\n", " ")}")


if __name__ == "__main__":
    main()
