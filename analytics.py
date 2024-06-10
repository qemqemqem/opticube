from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary
from pulp import *

import cvxpy as cp
import scipy.sparse as sp

from tqdm import tqdm
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
    row_avg_bullets = [f"- **{card}:** `{row_avg:.2f}`" for card, row_avg in top_n_row_cards]
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
    col_avg_bullets = [f"- **{card}:** `{col_avg:.2f}`" for card, col_avg in top_n_col_cards]
    console.print(Markdown("\n".join(col_avg_bullets)))


def analyze_synergy_symmetry(synergy_matrix, console):
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

def analyze_matrices(percentage_matrix, synergy_matrix, num_decks_matrix, card_names, card_name_to_id):
    console.print(Markdown("# Loaded Matrices"))
    percentage_avg_non_zero = percentage_matrix.sum() / percentage_matrix.nnz if percentage_matrix.nnz != 0 else 0
    console.print(Markdown(f"**Percentage Matrix:** Shape: `{percentage_matrix.shape}`, Non-zero entries: `{percentage_matrix.nnz}`, Average Non-zero Value: `{percentage_avg_non_zero:.2f}`"))

    synergy_avg_non_zero = synergy_matrix.sum() / synergy_matrix.nnz if synergy_matrix.nnz != 0 else 0
    console.print(Markdown(f"**Synergy Matrix:** Shape: `{synergy_matrix.shape}`, Non-zero entries: `{synergy_matrix.nnz}`, Average Non-zero Value: `{synergy_avg_non_zero:.2f}`"))

    num_decks_avg_non_zero = num_decks_matrix.sum() / num_decks_matrix.nnz if num_decks_matrix.nnz != 0 else 0
    console.print(Markdown(f"**Num Decks Matrix:** Shape: `{num_decks_matrix.shape}`, Non-zero entries: `{num_decks_matrix.nnz}`, Average Non-zero Value: `{num_decks_avg_non_zero:.2f}`"))

    # Ensure card_names is a list
    assert isinstance(card_names, list), "card_names should be a list."

    # Ensure card_name_to_id is a dictionary
    assert isinstance(card_name_to_id, dict), "card_name_to_id should be a dictionary."

    n = 3
    print_top_n(synergy_matrix, "Synergy Matrix", card_names, n=n)
    print_top_n(percentage_matrix, "Percentage Matrix", card_names, n=n)
    print_top_n(num_decks_matrix, "Num Decks Matrix", card_names, n=n)

    analyze_synergy_symmetry(synergy_matrix, console)

def find_good_subset_optimized_quadratic(matrix, set_size):
    # Error: cvxpy.error.SolverError: Problem is mixed-integer, but candidate QP/Conic solvers ([]) are not MIP-capable.

    # Ensure the matrix is symmetric
    if not (matrix != matrix.T).nnz == 0:  # Check if matrix is not symmetric
        matrix = (matrix + matrix.T) / 2  # Symmetrize the matrix
    n = matrix.shape[0]
    x = cp.Variable(n, boolean=True)
    objective = cp.Maximize(cp.quad_form(x, matrix))
    constraints = [cp.sum(x) == set_size]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)  # SCS is good for large-scale problems and supports sparse matrices
    return x.value

def find_good_subset_ilp(matrix, set_size):
    console.print(Markdown("# Finding Good Subset using ILP"))
    
    n = matrix.shape[0]
    prob = LpProblem("MaxSynergy", LpMaximize)
    choices = LpVariable.dicts("Choice", range(n), 0, 1, LpBinary)

    # Objective: We need to ensure that we're accessing matrix elements correctly and using them in a way that PuLP can handle.
    # Convert matrix to a list of lists if it's a numpy array to avoid indexing issues.
    if isinstance(matrix, np.ndarray):
        matrix = matrix.tolist()

    # Define the objective function
    prob += lpSum(matrix[i, j] * choices[i] * choices[j] for i in range(n) for j in range(n))

    # Constraint: Ensure the sum of choices equals the set size
    prob += lpSum(choices[i] for i in range(n)) == set_size

    # Solve the problem with a time limit
    prob.solve(PULP_CBC_CMD(timeLimit=60))  # Set a time limit of 60 seconds

    # Extract the solution
    solution_indices = [i for i in range(n) if choices[i].varValue == 1]
    return solution_indices

def find_good_subset(matrix, card_names, card_name_to_id, set_size: int, num_tries: int):
    console.print(Markdown("# Finding Good Subset"))
    best_goodness = -1
    best_subset = None

    for _ in tqdm(range(num_tries), desc="Finding best subset"):
        subset_indices = np.random.choice(matrix.shape[0], set_size, replace=False)
        goodness = 0

        for i in subset_indices:
            for j in subset_indices:
                if matrix[i, j] != 0:
                    goodness += matrix[i, j]

        if goodness > best_goodness:
            best_goodness = goodness
            best_subset = subset_indices

    best_subset_names = [card_names[i] for i in best_subset]
    best_subset_bullets = [f"- {name}" for name in best_subset_names]
    console.print(Markdown("**Best subset card names:**\n" + "\n".join(best_subset_bullets)))

    console.print(Markdown(f"**Goodness of best subset:** `{best_goodness}`"))

def main():
    percentage_matrix = load_percentage_matrix('percentage_matrix.pkl')
    synergy_matrix = load_percentage_matrix('synergy_matrix.pkl')
    num_decks_matrix = load_percentage_matrix('num_decks_matrix.pkl')

    with open('card_names.pkl', 'rb') as f:
        card_names = pickle.load(f)
    with open('card_name_to_id.pkl', 'rb') as f:
        card_name_to_id = pickle.load(f)

    analyze_matrices(percentage_matrix, synergy_matrix, num_decks_matrix, card_names, card_name_to_id)

    # find_good_subset(synergy_matrix, card_names, card_name_to_id, set_size=10, num_tries=1_000)

    # find_good_subset_optimized_quadratic(synergy_matrix, set_size=10)

    find_good_subset_ilp(synergy_matrix, set_size=10)

if __name__ == "__main__":
    main()