import pickle
import numpy as np

def load_percentage_matrix(file_path):
    with open(file_path, 'rb') as f:
        percentage_matrix = pickle.load(f)
    return percentage_matrix

def main():
    percentage_matrix = load_percentage_matrix('percentage_matrix.pkl')
    print(f"Loaded percentage matrix with shape: {percentage_matrix.shape} and {percentage_matrix.nnz} non-zero entries.")

    with open('card_names.pkl', 'rb') as f:
        card_names = pickle.load(f)
    with open('card_name_to_id.pkl', 'rb') as f:
        card_name_to_id = pickle.load(f)

    # Ensure card_names is a list
    assert isinstance(card_names, list), "card_names should be a list."

    # Ensure card_name_to_id is a dictionary
    assert isinstance(card_name_to_id, dict), "card_name_to_id should be a dictionary."

    # Calculate row sums
    row_sums = percentage_matrix.sum(axis=1)
    row_sums = np.array(row_sums)
    row_sums = row_sums.reshape(-1)
    top_10_row_indices = row_sums.argsort()[::-1][:10]
    top_10_row_cards = [(card_names[idx], row_sums[idx]) for idx in top_10_row_indices]
    print("Top 10 cards by row_sum:")
    for card, row_sum in top_10_row_cards:
        print(f"{card}: {row_sum}")

    # Calculate column sums
    col_sums = percentage_matrix.sum(axis=0)
    col_sums = np.array(col_sums).flatten()
    top_10_col_indices = col_sums.argsort()[::-1][:10]
    top_10_col_cards = [(card_names[idx], col_sums[idx]) for idx in top_10_col_indices]
    print("Top 10 cards by col_sum:")
    for card, col_sum in top_10_col_cards:
        print(f"{card}: {col_sum}")

    # Calculate synergy symmetry
    asymmetry_count = 0
    total_difference = 0
    count_difference = 0

    for i in range(percentage_matrix.shape[0]):
        for j in range(percentage_matrix.shape[1]):
            if percentage_matrix[i, j] != 0 and percentage_matrix[j, i] == 0:
                asymmetry_count += 1
            elif percentage_matrix[i, j] != 0 and percentage_matrix[j, i] != 0:
                total_difference += abs(percentage_matrix[i, j] - percentage_matrix[j, i])
                count_difference += 1

    average_difference = total_difference / count_difference if count_difference != 0 else 0

    print(f"Asymmetry count (X,Y exists but Y,X does not): {asymmetry_count}")
    print(f"Average difference where both entries exist: {average_difference}")

if __name__ == "__main__":
    main()