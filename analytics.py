import pickle
import numpy as np
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from scipy.sparse import issparse

console = Console(width=100, soft_wrap=True)

def load_percentage_matrix(file_path):
    with open(file_path, 'rb') as f:
        percentage_matrix = pickle.load(f)
    return percentage_matrix

def print_top_n(matrix, title, card_names, n=10):
    console.print(Markdown(f"# {title}"))

    console.print(Markdown(f"## Top {n} Cards by {title} Row Sum"))
    row_sums = matrix.sum(axis=1)
    row_sums = np.array(row_sums).reshape(-1)
    top_n_row_indices = row_sums.argsort()[::-1][:n]
    top_n_row_cards = [(card_names[idx], row_sums[idx]) for idx in top_n_row_indices]
    row_sum_bullets = [f"- **{card}:** `{row_sum}`" for card, row_sum in top_n_row_cards]
    console.print(Markdown("\n".join(row_sum_bullets)))

    console.print(Markdown(f"## Top {n} Cards by {title} Row Average"))
    # Calculate row averages considering only non-zero entries
    row_counts = matrix.getnnz(axis=1)  # getnnz counts the number of non-zero entries along the given axis, assuming that it's sparse
    row_counts += 1  # Add 1 as an epsilon to avoid division by zero
    row_averages = row_sums / row_counts
    row_averages = np.array(row_averages).reshape(-1)
    top_n_row_indices = row_averages.argsort()[::-1][:n]
    top_n_row_cards = [(card_names[idx], row_averages[idx]) for idx in top_n_row_indices]
    row_avg_bullets = [f"- **{card}:** `{row_avg}`" for card, row_avg in top_n_row_cards]
    console.print(Markdown("\n".join(row_avg_bullets)))

    console.print(Markdown(f"## Top {n} Cards by {title} Column Sum"))
    # Calculate column sums
    col_sums = matrix.sum(axis=0)
    col_sums = np.array(col_sums).flatten()
    top_n_col_indices = col_sums.argsort()[::-1][:n]
    top_n_col_cards = [(card_names[idx], col_sums[idx]) for idx in top_n_col_indices]
    col_sum_bullets = [f"- **{card}:** `{col_sum}`" for card, col_sum in top_n_col_cards]
    console.print(Markdown("\n".join(col_sum_bullets)))

    console.print(Markdown(f"## Top {n} Cards by {title} Column Average"))
    # Calculate column averages considering only non-zero entries
    col_counts = matrix.getnnz(axis=0)  # getnnz counts the number of non-zero entries along the given axis, assuming that it's sparse
    col_counts += 1  # Add 1 as an epsilon to avoid division by zero
    col_averages = col_sums / col_counts
    col_averages = np.array(col_averages).flatten()
    top_n_col_indices = col_averages.argsort()[::-1][:n]
    top_n_col_cards = [(card_names[idx], col_averages[idx]) for idx in top_n_col_indices]
    col_avg_bullets = [f"- **{card}:** `{col_avg}`" for card, col_avg in top_n_col_cards]
    console.print(Markdown("\n".join(col_avg_bullets)))
def analyze_matrices(percentage_matrix, synergy_matrix, num_decks_matrix, card_names, card_name_to_id):
    console.print(Markdown("# Loaded Matrices"))
    console.print(Markdown(f"**Percentage Matrix Shape:** `{percentage_matrix.shape}` **Non-zero entries:** `{percentage_matrix.nnz}`"))
    console.print(Markdown(f"**Synergy Matrix Shape:** `{synergy_matrix.shape}` **Non-zero entries:** `{synergy_matrix.nnz}`"))
    console.print(Markdown(f"**Num Decks Matrix Shape:** `{num_decks_matrix.shape}` **Non-zero entries:** `{num_decks_matrix.nnz}`"))

    # Ensure card_names is a list
    assert isinstance(card_names, list), "card_names should be a list."

    # Ensure card_name_to_id is a dictionary
    assert isinstance(card_name_to_id, dict), "card_name_to_id should be a dictionary."

    print_top_10(synergy_matrix, "Synergy Matrix", card_names)
    # print_top_10(percentage_matrix, "Percentage Matrix", card_names)
    # print_top_10(num_decks_matrix, "Num Decks Matrix", card_names)

    console.print(Markdown("## Synergy Symmetry Analysis"))
    # Calculate synergy symmetry
    sample_size = 10000
    rows, cols = synergy_matrix.shape

    # Generate random indices for rows and columns
    random_rows = np.random.choice(rows, sample_size, replace=True)
    random_cols = np.random.choice(cols, sample_size, replace=True)

    asymmetry_count = 0
    total_difference = 0
    count_difference = 0
    both_zero_count = 0
    both_exist_count = 0

    for i, j in zip(random_rows, random_cols):
        if synergy_matrix[i, j] != 0 and synergy_matrix[j, i] == 0:
            asymmetry_count += 1
        elif synergy_matrix[i, j] == 0 and synergy_matrix[j, i] != 0:
            asymmetry_count += 1
        elif synergy_matrix[i, j] != 0 and synergy_matrix[j, i] != 0:
            total_difference += abs(synergy_matrix[i, j] - synergy_matrix[j, i])
            count_difference += 1
            both_exist_count += 1
        elif synergy_matrix[i, j] == 0 and synergy_matrix[j, i] == 0:
            both_zero_count += 1
    average_difference = total_difference / count_difference if count_difference != 0 else 0

    console.print(Markdown(f"**Asymmetry count (X,Y exists but Y,X does not):** `{asymmetry_count}`"))
    console.print(Markdown(f"**Average difference where both entries exist:** `{average_difference}`"))
    console.print(Markdown(f"**Both zero count:** `{both_zero_count}`"))
    console.print(Markdown(f"**Both exist count:** `{both_exist_count}`"))

def main():
    percentage_matrix = load_percentage_matrix('percentage_matrix.pkl')
    synergy_matrix = load_percentage_matrix('synergy_matrix.pkl')
    num_decks_matrix = load_percentage_matrix('num_decks_matrix.pkl')

    with open('card_names.pkl', 'rb') as f:
        card_names = pickle.load(f)
    with open('card_name_to_id.pkl', 'rb') as f:
        card_name_to_id = pickle.load(f)

    analyze_matrices(percentage_matrix, synergy_matrix, num_decks_matrix, card_names, card_name_to_id)
if __name__ == "__main__":
    main()