
import numpy as np
import scipy.sparse as sp


def main():
    card_names = []
    card_name_to_id = {}
    with seen_cards_file.open('r') as f:
        for idx, line in enumerate(f):
            card_name, _ = line.strip().split(',')
            card_names.append(card_name)
            card_name_to_id[card_name] = idx

    size = len(card_names)

    # Create an empty lil_matrix
    lil_matrix = sp.lil_matrix((size, size))

    # DEMO

    # Number of random entries to insert
    num_entries = 1000

    # Populate the matrix with random numbers in random cells
    for _ in range(num_entries):
        row = np.random.randint(0, size)
        col = np.random.randint(0, size)
        value = np.random.rand()
        
        lil_matrix[row, col] = value

    # Convert the lil_matrix to a csr_matrix for efficient operations
    csr_matrix = lil_matrix.tocsr()

    # Print some information about the matrix
    print(f"Shape of the matrix: {csr_matrix.shape}")
    print(f"Number of non-zero entries: {csr_matrix.nnz}")

    # Optionally, print the non-zero entries
    print("Non-zero entries:")
    print(csr_matrix)

if __name__ == "__main__":
    main()

