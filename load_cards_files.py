
import pickle
import numpy as np
import scipy.sparse as sp
from utils import format_card_name


def main():
    card_names = []
    card_name_to_id = {}
    with open('seen_cards.txt', 'r') as f:
        for idx, line in enumerate(f):
            card_name, _ = line.strip().split(',')
            card_names.append(card_name)
            card_name_to_id[card_name] = idx
    size = len(card_names)

    # Save card_names and card_name_to_id to file
    with open('card_names.pkl', 'wb') as f:
        pickle.dump(card_names, f)
    with open('card_name_to_id.pkl', 'wb') as f:
        pickle.dump(card_name_to_id, f)

    # Create empty lil_matrices
    synergy_matrix = sp.lil_matrix((size, size))
    percentage_matrix = sp.lil_matrix((size, size))
    num_decks_matrix = sp.lil_matrix((size, size))

    from pathlib import Path
    from tqdm import tqdm

    cards_dir = Path('cards')

    # Iterate over each file in the cards directory
    total_files = sum(1 for _ in cards_dir.iterdir())
    for card_file in tqdm(cards_dir.iterdir(), desc="Processing card files", total=total_files):
        card_name_outer = format_card_name(card_file.stem)  # Get the formatted card name from the file name (without extension)
        if card_name_outer in card_name_to_id:
            card_id_outer = card_name_to_id[card_name_outer]
            with card_file.open('r') as file:
                for line in file:
                    try:
                        # Example line:
                        # Satyr Wayfinder: 23% of 4829 decks +22% synergy
                        card_name, rest = line.split(':', 1)
                        card_id_inner = card_name_to_id[format_card_name(card_name.strip())]

                        synergy_percentage = float(rest.split('synergy')[0].split()[-1].replace('%', ''))
                        synergy_matrix[card_id_outer, card_id_inner] = synergy_percentage

                        # In the part of the line like "22% of 4829 decks", extract both numbers
                        percentage, num_decks = rest.split('of')
                        percentage = float(percentage.strip().replace('%', ''))
                        num_decks = int(num_decks.strip().split()[0])
                        
                        # Fill the percentage_matrix and num_decks_matrix
                        percentage_matrix[card_id_outer, card_id_inner] = percentage
                        num_decks_matrix[card_id_outer, card_id_inner] = num_decks
                    except ValueError:
                        pass
                        # print(f"Error processing line: {line}")
                    except KeyError:
                        pass
                        # print(f"Card name not found: {format_card_name(card_name.strip())}")

    # Convert the lil_matrix to a csr_matrix for efficient operations
    synergy_matrix = synergy_matrix.tocsr()
    percentage_matrix = percentage_matrix.tocsr()
    num_decks_matrix = num_decks_matrix.tocsr()

    # Print some information about the matrices
    print(f"Synergy Matrix - Shape: {synergy_matrix.shape}, Non-zero entries: {synergy_matrix.nnz}, Percent non-empty: {synergy_matrix.nnz / (synergy_matrix.shape[0] * synergy_matrix.shape[1]) * 100:.2f}%")
    print(f"Percentage Matrix - Shape: {percentage_matrix.shape}, Non-zero entries: {percentage_matrix.nnz}, Percent non-empty: {percentage_matrix.nnz / (percentage_matrix.shape[0] * percentage_matrix.shape[1]) * 100:.2f}%")
    print(f"Number of Decks Matrix - Shape: {num_decks_matrix.shape}, Non-zero entries: {num_decks_matrix.nnz}, Percent non-empty: {num_decks_matrix.nnz / (num_decks_matrix.shape[0] * num_decks_matrix.shape[1]) * 100:.2f}%")

    # Save synergy_matrix to file
    with open('synergy_matrix.pkl', 'wb') as f:
        pickle.dump(synergy_matrix, f)
    
    with open('percentage_matrix.pkl', 'wb') as f:
        pickle.dump(percentage_matrix, f)
    
    with open('num_decks_matrix.pkl', 'wb') as f:
        pickle.dump(num_decks_matrix, f)


if __name__ == "__main__":
    main()

