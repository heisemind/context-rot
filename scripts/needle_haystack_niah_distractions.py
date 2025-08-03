from typing import List, Dict, Optional, Any, Tuple
from langchain_google_vertexai import ChatVertexAI
from time import sleep
import pandas as pd
import random
import os
import re

SHUFFLE_HAYSTACK = False
LLM_MODEL = "gpt-3.5-turbo"

def load_story_with_needle_and_distractor(story_path: str, needle: str, distractors: list[str]) -> list[str]:
    """
    Load a story from a text file, split it into phrases, and insert the needle and a random distractor at random positions.
    Args:
        story_path (str): Path to the story text file.
        needle (str): The phrase to be inserted as the needle.
        distractors (list[str]): List of possible distractor phrases to insert.
    Returns:
        list[str]: List of phrases with the needle and one random distractor inserted at random positions.
    Details:
        - Reads the story from the given file path.
        - Splits the story into phrases using periods as delimiters.
        - Removes empty phrases and trims whitespace.
        - Inserts the needle and a randomly chosen distractor at two random positions in the phrase list.
    """

    with open(story_path, "r", encoding="utf-8") as f:
        story_text = f.read()
    phrases = re.split(r"\.\s+|\.$", story_text)
    phrases = [p.strip() for p in phrases if p.strip()]
    insert_positions = sorted(random.sample(range(len(phrases)), 2))
    phrases.insert(insert_positions[0], needle)
    phrases.insert(insert_positions[1]+1, random.choice(distractors))
    return phrases

def load_haystack_phrases(needle: str, distractors: List[str]) -> List[str]:
    """
    Load haystack phrases based on shuffle_haystack argument.
    If True, load instances from the largest cluster in lovecraft_final_clusters.csv.
    If False, load phrases from the_colour_out_of_space.txt.
    Args:
        needle: The needle phrase to insert.
        distractors: List of distractor phrases.
    Returns:
        List of phrases with needle and distractor inserted.
    """
    if SHUFFLE_HAYSTACK:
        cluster_df = pd.read_csv("data/lovecraft_final_clusters.csv")
        largest_cluster = cluster_df['cluster'].value_counts().idxmax()
        cluster_phrases = cluster_df[cluster_df['cluster'] == largest_cluster]['phrase'].tolist()
        # Remove the needle if present
        cluster_phrases = [p for p in cluster_phrases if p != needle]
        # Shuffle and select a subset
        random.shuffle(cluster_phrases)
        phrases = cluster_phrases[:100]  # Arbitrary size, adjust as needed
    else:
        with open("data/the_colour_out_of_space.txt", "r", encoding="utf-8") as f:
            story_text = f.read()
        phrases = re.split(r"\.\s+|\.$", story_text)
        phrases = [p.strip() for p in phrases if p.strip()]
    # Insert needle and a random distractor at random positions
    insert_positions = sorted(random.sample(range(len(phrases)), 2))
    phrases.insert(insert_positions[0], needle)
    phrases.insert(insert_positions[1]+1, random.choice(distractors))
    return phrases

# Setup experiment data: load haystack phrases, cluster assignments, and needle cluster
def setup_experiment(needle: str, distractors: List[str]) -> Tuple[List[str], pd.DataFrame, Any]:
    """
    Prepare experiment data: load haystack phrases, cluster assignments, and needle cluster.
    Args:
        needle: The needle phrase.
        distractors: List of distractor phrases.
    Returns:
        Tuple of (phrases, cluster_df, needle_cluster)
    """
    phrases = load_haystack_phrases(needle, distractors)
    cluster_df = pd.read_csv("data/lovecraft_final_clusters.csv")
    needle_cluster = cluster_df[cluster_df['phrase'] == needle]['cluster'].values[0] if not cluster_df[cluster_df['phrase'] == needle].empty else None
    random.seed(42)
    return phrases, cluster_df, needle_cluster

# Tokenizer function (approximate, for OpenAI models)
def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text using whitespace splitting.
    Args:
        text: Input text string.
    Returns:
        Number of tokens (int).
    """
    # Simple whitespace tokenization as a proxy
    return len(text.split())


def add_distractors(
    context_phrases: List[str],
    n_distractors: int,
    distractors: List[str],
    needle_idx: Optional[int] = None
) -> List[str]:
    """
    Insert distractor phrases into context at random positions, avoiding the needle index.
    Args:
        context_phrases (List[str]): List of context phrases.
        n_distractors (int): Number of distractors to add.
        distractors (List[str]): List of distractor phrases.
        needle_idx (Optional[int]): Index of the needle phrase to avoid.
    Returns:
        List[str]: Modified list of context phrases with distractors added.
    """
    context_len = len(context_phrases)
    available_positions = [i for i in range(context_len) if i != needle_idx]
    positions = random.sample(available_positions, n_distractors)
    distractors_sample = random.sample(distractors, n_distractors)
    for distractor, pos_idx in zip(distractors_sample, positions):
        context_phrases.insert(pos_idx, distractor)
    return context_phrases

def get_context_list(
    needle: str,
    phrases: List[str],
    n_tokens: int,
    cluster_df: Optional[pd.DataFrame],
    needle_cluster: Optional[Any]
) -> List[str]:
    """
    Build a list of context phrases up to n_tokens, optionally prioritizing cluster phrases.
    Args:
        needle (str): The needle phrase to exclude from context.
        phrases (List[str]): List of all available phrases.
        n_tokens (int): Target number of tokens in context.
        cluster_df (Optional[pd.DataFrame]): DataFrame of cluster assignments.
        needle_cluster (Optional[Any]): Cluster id for needle.
    Returns:
        List[str]: List of context phrases up to n_tokens.
    Details:
        - If cluster_df and needle_cluster are provided, prioritizes phrases from the same cluster as the needle.
        - Fills up to n_tokens with additional phrases if needed.
        - Excludes the needle phrase from the context.
    """
    if cluster_df is not None and needle_cluster is not None:
        cluster_phrases = cluster_df[cluster_df['cluster'] == needle_cluster]['phrase'].tolist()
        cluster_phrases = [p for p in cluster_phrases if p != needle]
        context_list = cluster_phrases.copy()
        if len(context_list) < n_tokens:
            other_phrases = [p for p in phrases if p not in context_list and p != needle]
            context_list += other_phrases
    else:
        context_list = [p for p in phrases if p != needle]
    # Build context up to n_tokens
    context = ""
    context_tokens = 0
    idx = 0
    while context_tokens < n_tokens and idx < len(context_list):
        context += context_list[idx] + " "
        context_tokens = count_tokens(context)
        idx += 1
    return context.strip().split(' ')

def insert_needle(
    context_phrases: List[str],
    needle: str,
    pos: Optional[int]
) -> Tuple[List[str], int]:
    """
    Insert the needle phrase into the context phrases at a region defined by pos.
    Args:
        context_phrases (List[str]): List of context phrases.
        needle (str): The needle phrase to insert.
        pos (Optional[int]): Region index for needle placement (0-4 for 5 regions, or None for start).
    Returns:
        Tuple[List[str], int]: Modified context phrases and the index where the needle was inserted.
    Details:
        - If pos is provided, splits context into 5 regions and inserts needle randomly in the specified region.
        - If pos is None, inserts needle at the start.
    """
    if pos is not None:
        context_len = len(context_phrases)
        region_size = max(1, context_len // 5)
        start = pos * region_size
        end = min(start + region_size, context_len)
        insert_idx = random.randint(start, end - 1)
        context_phrases.insert(insert_idx, needle)
    else:
        insert_idx = 0
        context_phrases.insert(0, needle)
    return context_phrases, insert_idx

def build_context(
    needle: str,
    phrases: List[str],
    n_tokens: int,
    pos: Optional[int],
    cluster_df: Optional[pd.DataFrame],
    needle_cluster: Optional[Any],
    n_distractors: int,
    distractors: List[str]
) -> Tuple[str, int]:
    """
    Build a context string for the LLM experiment, containing the needle and a specified number of distractors.
    Args:
        needle (str): The phrase to be inserted as the needle in the context.
        phrases (List[str]): List of all available context phrases.
        n_tokens (int): Target number of tokens in the context.
        pos (Optional[int]): Region index for needle placement (0-4 for 5 regions, or None for start).
        cluster_df (Optional[pd.DataFrame]): DataFrame of cluster assignments (if clustering is used).
        needle_cluster (Optional[Any]): Cluster id for the needle (if clustering is used).
        n_distractors (int): Number of distractor phrases to insert into the context.
        distractors (List[str]): List of possible distractor phrases to insert.
    Returns:
        Tuple[str, int]:
            - context (str): The final context string containing the needle and distractors.
            - final_tokens (int): The total number of tokens in the context string.
    Details:
        - Builds a list of context phrases up to n_tokens, prioritizing cluster phrases if provided.
        - Inserts the needle at a random position within the specified region.
        - Inserts the specified number of distractors at random positions, avoiding the needle index.
        - Joins the context phrases into a single string and counts the tokens.
    """
    context_phrases = get_context_list(needle, phrases, n_tokens, cluster_df, needle_cluster)
    context_phrases, insert_idx = insert_needle(context_phrases, needle, pos)
    if n_distractors > 0 and distractors is not None:
        context_phrases = add_distractors(context_phrases, n_distractors, distractors, needle_idx=insert_idx)
    final_context = ' '.join(context_phrases)
    final_tokens = count_tokens(final_context)
    return final_context, final_tokens

def query_llm(context: str, question: str, model: str = "gemini-2.5-flash") -> str:
    """
    Query the LLM (VertexAI for Gemini models, OpenAI otherwise) with the given context and question.
    Args:
        context: The context string.
        question: The question string.
        model: Model name to use.
    Returns:
        The answer string from the LLM.
    """
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nEnsure to respond only and exactly the name asked in the question. Answer:"
    sleep(0.5)
    if "gemini" in model:
        llm = ChatVertexAI(
            model_name=model,
            temperature=0,
            max_output_tokens=512,
        )
        response = llm.invoke(prompt)
        return response.content
    else:
        from openai import OpenAI, RateLimitError
        openai = OpenAI()
        while True:
            try:
                response = openai.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content.strip()
            except RateLimitError:
                print("Rate limit exceeded. Waiting 3 seconds before retrying...")
                sleep(3)

def run_experiment(
    needle: str,
    question: str,
    phrases: List[str],
    steps: List[int],
    model: str,
    cluster_df: Optional[pd.DataFrame]=None,
    needle_cluster: Optional[Any]=None
) -> List[Dict[str, Any]]:
    """
    Run the experiment for all combinations of tokens, positions, and distractors.
    Args:
        needle: The needle phrase.
        question: The question string.
        phrases: List of all phrases.
        steps: List of token counts to test.
        model: Model name to use.
        cluster_df: DataFrame of cluster assignments (optional).
        needle_cluster: Cluster id for needle (optional).
    Returns:
        List of result dictionaries for each run.
    """
    results = []
    distractor_counts = [0, 2, 4, 8, 16]
    for n_tokens in steps:
        for pos in range(5):
            for n_distractors in distractor_counts:
                context, context_tokens = build_context(needle, phrases, n_tokens, pos, cluster_df, needle_cluster, n_distractors)
                answer = query_llm(context, question, model)
                results.append({
                    "n_tokens": context_tokens,
                    "needle_position": pos,
                    "n_distractors": n_distractors,
                    "answer": answer
                })
                print(f"Tokens: {context_tokens}, Needle Position: {pos}, Distractors: {n_distractors}, Answer: {answer}")
    return results

def get_missing_combos(
    steps: List[int],
    distractor_counts: List[int],
    prev_results: pd.DataFrame
) -> pd.DataFrame:
    """
    Build all possible combinations and return missing ones.
    Args:
        steps (List[int]): List of token counts to test.
        distractor_counts (List[int]): List of distractor counts to test.
        prev_results (pd.DataFrame): DataFrame of previous results.
    Returns:
        pd.DataFrame: DataFrame of missing combinations to run.
    Details:
        - Generates all combinations of n_tokens, needle_position, n_distractors.
        - Returns only those not present in prev_results.
    """
    all_combos = pd.DataFrame([
        (n_tokens, pos, n_distractors)
        for n_tokens in steps
        for pos in range(5)
        for n_distractors in distractor_counts
    ], columns=["n_tokens", "needle_position", "n_distractors"])
    if not prev_results.empty:
        merged = all_combos.merge(prev_results, on=["n_tokens", "needle_position", "n_distractors"], how="left", indicator=True)
        missing_combos = merged[merged["_merge"] == "left_only"][["n_tokens", "needle_position", "n_distractors"]]
    else:
        missing_combos = all_combos
    return missing_combos

def run_and_save_missing_combos(
    missing_combos: pd.DataFrame,
    needle: str,
    phrases: List[str],
    cluster_df: pd.DataFrame,
    needle_cluster: Any,
    question: str,
    model: str,
    prev_results: pd.DataFrame,
    result_path: str,
    distractors: List[str]
) -> None:
    """
    Run only missing experiment combinations and save results after every execution.
    Args:
        missing_combos (pd.DataFrame): DataFrame of missing experiment combinations.
        needle (str): The needle phrase.
        phrases (List[str]): List of all context phrases.
        cluster_df (pd.DataFrame): DataFrame of cluster assignments.
        needle_cluster (Any): Cluster id for needle.
        question (str): The question to ask the LLM.
        model (str): Model name to use.
        prev_results (pd.DataFrame): DataFrame of previous results.
        result_path (str): Path to save results CSV.
        distractors (List[str]): List of distractor phrases.
    Returns:
        None
    Details:
        - Iterates over all missing experiment combinations.
        - Builds context and queries the LLM for each combination.
        - Appends results and saves to CSV after each run.
        - Prints progress and results to stdout.
    """
    new_results = []
    for _, row in missing_combos.iterrows():
        context, context_tokens = build_context(
            needle, phrases, row["n_tokens"], row["needle_position"], cluster_df, needle_cluster, row["n_distractors"], distractors)
        answer = query_llm(context, question, model)
        run_result = {
            "n_tokens": row["n_tokens"],
            "needle_position": row["needle_position"],
            "n_distractors": row["n_distractors"],
            "answer": answer
        }
        new_results.append(run_result)
        combined = pd.concat([prev_results, pd.DataFrame(new_results)], ignore_index=True)
        combined.to_csv(result_path, index=False)
        print(f"Tokens: {row['n_tokens']}, Needle Position: {row['needle_position']}, Distractors: {row['n_distractors']}, Answer: {answer}")
    if not new_results:
        print("No missing combinations to run.")

def main() -> None:
    """
    Main entry point for running the needle haystack experiment.
    Sets up experiment parameters, loads data, finds missing combinations, and runs the experiment.
    Returns:
        None
    """
    needle = "The book hidden beneath the gnarled oak in the blighted landscape was titled Alhazred’s Codex."
    question = "What was the name of the book hidden beneath the gnarled oak in the blighted landscape described in the text?"
    distractors = [
        "A scroll was hidden beneath the gnarled oak in the blighted landscape, inscribed with ancient runes.",
        "The book hidden beneath the gnarled oak in the cursed forest was titled The Grimoire of Shadows.",
        "A tome was hidden beneath a withered elm in the blighted landscape, inscribed with the name The Codex of Eternity.",
        "Beneath the gnarled oak in the blighted landscape, a map was found etched with the name Vossoth’s Path.",
        "The book hidden beneath the gnarled oak in a desolate wasteland was called The Tome of the Void.",
        "A chest hidden beneath the gnarled oak in the blighted landscape contained artifacts inscribed with Vossoth’s Legacy.",
        "The book hidden beneath the roots of a gnarled oak in the shadowed valley was titled The Writings of Vossoth.",
        "A stone tablet was hidden beneath the gnarled oak in the blighted landscape, carved with the name Vossoth’s Prophecy.",
        "The book hidden beneath the gnarled oak in the blighted landscape was written by Vossoth.",
        "A book was hidden beneath a gnarled oak in the blighted hills, inscribed with The Codex of the Damned.",
        "Beneath the gnarled oak in the blighted landscape, a journal was found inscribed with Vossoth’s Notes.",
        "The book hidden beneath the gnarled oak in a ruined landscape was titled The Tome of Forgotten Lore.",
        "A relic was hidden beneath the gnarled oak in the blighted landscape, engraved with Vossoth’s Sigil.",
        "The book hidden beneath the roots of a gnarled oak in the blighted marshes was called The Codex of Mysteries.",
        "Beneath the gnarled oak in the blighted landscape, a parchment was found inscribed with The Oath of Vossoth.",
        "The map hidden beneath the gnarled oak in the blighted landscape was burned and its title was illegible.",
        "A diary was hidden beneath the gnarled oak in the blighted landscape, its cover marked with the sigil of Vossoth."
    ]
    phrases, cluster_df, needle_cluster = setup_experiment(needle, distractors)
    random.seed(42)
    steps = [100, 256, 1024, 2048, 4096, 8192, 10000, 12000]
    result_path = f"results/needle_haystack_niah_{LLM_MODEL}.csv"
    distractor_counts = [0, 2, 4, 8, 16]
    if os.path.exists(result_path):
        prev_results = pd.read_csv(result_path)
    else:
        prev_results = pd.DataFrame()
    missing_combos = get_missing_combos(steps, distractor_counts, prev_results)
    run_and_save_missing_combos(missing_combos, needle, phrases, cluster_df, needle_cluster, question, LLM_MODEL, prev_results, result_path, distractors)

if __name__ == "__main__":
    main()