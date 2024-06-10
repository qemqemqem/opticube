import pickle
import numpy as np
from rich import print
from rich.console import Console
from rich.markdown import Markdown

console = Console(width=100, soft_wrap=True)

def load_percentage_matrix(file_path):
    with open(file_path, 'rb') as f:
        percentage_matrix = pickle.load(f)
    return percentage_matrix

def main():
    percentage_matrix = load_percentage_matrix('percentage_matrix.pkl')
    console.print(Markdown("# Loaded Percentage Matrix"))
    console.print(Markdown(f"**Shape:** `{percentage_matrix.shape}` **Non-zero entries:** `{percentage_matrix.nnz}`"))

    with open('card_names.pkl', 'rb') as f:
        card_names = pickle.load(f)
    with open('card_name_to_id.pkl', 'rb') as f:
        card_name_to_id = pickle.load(f)

    # Ensure card_names is a list
    assert isinstance(card_names, list), "card_names should be a list."

    # Ensure card_name_to_id is a dictionary
    assert isinstance(card_name_to_id, dict), "card_name_to_id should be a dictionary."

    console.print(Markdown("## Top 10 Cards by Row Average"))
    # Calculate row averages
    row_averages = percentage_matrix.mean(axis=1)
    row_averages = np.array(row_averages)
    row_averages = row_averages.reshape(-1)
    top_10_row_indices = row_averages.argsort()[::-1][:10]
    top_10_row_cards = [(card_names[idx], row_averages[idx]) for idx in top_10_row_indices]
    row_avg_bullets = [f"- **{card}:** `{row_avg}`" for card, row_avg in top_10_row_cards]
    console.print(Markdown("\n".join(row_avg_bullets)))

    console.print(Markdown("## Top 10 Cards by Column Average"))
    # Calculate column averages
    col_averages = percentage_matrix.mean(axis=0)
    col_averages = np.array(col_averages).flatten()
    top_10_col_indices = col_averages.argsort()[::-1][:10]
    top_10_col_cards = [(card_names[idx], col_averages[idx]) for idx in top_10_col_indices]
    col_avg_bullets = [f"- **{card}:** `{col_avg}`" for card, col_avg in top_10_col_cards]
    console.print(Markdown("\n".join(col_avg_bullets)))

    console.print(Markdown("## Synergy Symmetry Analysis"))
    # Calculate synergy symmetry
    sample_size = 10000
    rows, cols = percentage_matrix.shape

    # Generate random indices for rows and columns
    random_rows = np.random.choice(rows, sample_size, replace=True)
    random_cols = np.random.choice(cols, sample_size, replace=True)

    asymmetry_count = 0
    total_difference = 0
    count_difference = 0
    both_zero_count = 0
    both_exist_count = 0

    for i, j in zip(random_rows, random_cols):
        if percentage_matrix[i, j] != 0 and percentage_matrix[j, i] == 0:
            asymmetry_count += 1
        elif percentage_matrix[i, j] == 0 and percentage_matrix[j, i] != 0:
            asymmetry_count += 1
        elif percentage_matrix[i, j] != 0 and percentage_matrix[j, i] != 0:
            total_difference += abs(percentage_matrix[i, j] - percentage_matrix[j, i])
            count_difference += 1
            both_exist_count += 1
        elif percentage_matrix[i, j] == 0 and percentage_matrix[j, i] == 0:
            both_zero_count += 1
    average_difference = total_difference / count_difference if count_difference != 0 else 0

    console.print(Markdown(f"**Asymmetry count (X,Y exists but Y,X does not):** `{asymmetry_count}`"))
    console.print(Markdown(f"**Average difference where both entries exist:** `{average_difference}`"))
    console.print(Markdown(f"**Both zero count:** `{both_zero_count}`"))
    console.print(Markdown(f"**Both exist count:** `{both_exist_count}`"))

if __name__ == "__main__":
    main()