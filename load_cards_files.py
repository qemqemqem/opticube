"""
Step 4: Build sparse matrices from EDHREC JSON data.

Reads cards_json/*.json files and constructs:
- synergy_matrix: pairwise synergy scores (S[i,j] = synergy of card j on card i's page)
- lift_matrix: statistical lift values
- inclusion_matrix: number of decks containing both cards
- potential_decks_matrix: total reference population size

Also symmetrizes the synergy matrix: S' = (S + S^T) / 2 where both directions
exist, keeping the single direction where only one exists.

Outputs are saved as .pkl files alongside card_names.pkl and card_name_to_id.pkl.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from utils import format_card_name

CARDS_JSON_DIR = Path("cards_json")
SEEN_CARDS_FILE = Path("seen_cards.txt")


def extract_card_pairs(json_data: dict) -> list[dict]:
    """
    Extract all card pair entries from an EDHREC JSON response.

    Returns a list of dicts with keys: sanitized, name, synergy, lift,
    inclusion, num_decks, potential_decks.
    """
    pairs = []

    # The synergy data lives at container.json_dict.cardlists
    json_dict = json_data.get("container", {}).get("json_dict", {})
    cardlists = json_dict.get("cardlists", [])

    for cardlist in cardlists:
        tag = cardlist.get("tag", "")

        # Skip commander lists -- those are a different kind of relationship
        if tag == "topcommanders":
            continue

        cardviews = cardlist.get("cardviews", [])
        for card in cardviews:
            synergy = card.get("synergy")
            if synergy is None:
                continue

            pairs.append({
                "sanitized": card.get("sanitized", ""),
                "name": card.get("name", ""),
                "synergy": float(synergy),
                "lift": float(card.get("lift", 0)),
                "inclusion": int(card.get("inclusion", 0)),
                "num_decks": int(card.get("num_decks", 0)),
                "potential_decks": int(card.get("potential_decks", 0)),
            })

    return pairs


def build_card_index(pool_size: int = None) -> tuple[list[str], dict[str, int]]:
    """
    Build the card name list and name-to-ID mapping from seen_cards.txt.

    Deduplicates by card name (keeping highest count), then sorts by frequency.
    This ensures consistent indexing across all matrices.

    Args:
        pool_size: If set, only include the top N cards by frequency.
                   If None, include all cards from seen_cards.txt.

    Returns:
        (card_names, card_name_to_id): List of card names and mapping from name to index.
    """
    card_counts: dict[str, int] = {}

    with SEEN_CARDS_FILE.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            card_name = parts[0].strip()
            count = int(parts[1].strip())
            if card_name not in card_counts or count > card_counts[card_name]:
                card_counts[card_name] = count

    # Sort by frequency descending (consistent ordering)
    sorted_cards = sorted(card_counts.items(), key=lambda x: x[1], reverse=True)

    # Limit to pool_size if specified
    if pool_size is not None:
        sorted_cards = sorted_cards[:pool_size]

    card_names = []
    card_name_to_id = {}
    for idx, (name, _count) in enumerate(sorted_cards):
        card_names.append(name)
        card_name_to_id[name] = idx

    return card_names, card_name_to_id


def build_matrices(pool_size: int = None):
    """
    Build all sparse matrices from the downloaded JSON data.

    Args:
        pool_size: If set, only build matrices for the top N cards by frequency.
                   If None, build matrices for all cards in seen_cards.txt.
    """
    # Build the card index
    card_names, card_name_to_id = build_card_index(pool_size=pool_size)
    size = len(card_names)
    if pool_size is not None:
        print(f"Card pool size: {size} cards (top {pool_size} from {SEEN_CARDS_FILE})")
    else:
        print(f"Card universe size: {size} cards (from {SEEN_CARDS_FILE})")

    # Save card index
    with open("card_names.pkl", "wb") as f:
        pickle.dump(card_names, f)
    with open("card_name_to_id.pkl", "wb") as f:
        pickle.dump(card_name_to_id, f)

    # Create empty lil_matrices (efficient for incremental construction)
    synergy_matrix = sp.lil_matrix((size, size))
    lift_matrix = sp.lil_matrix((size, size))
    inclusion_matrix = sp.lil_matrix((size, size))
    potential_decks_matrix = sp.lil_matrix((size, size))

    # Count JSON files
    json_files = list(CARDS_JSON_DIR.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {CARDS_JSON_DIR}")

    if not json_files:
        print("ERROR: No JSON files found. Run scrape_edhrec.py first.")
        return

    total_pairs = 0
    skipped_unknown = 0

    for json_file in tqdm(json_files, desc="Processing JSON files"):
        outer_name = json_file.stem  # e.g., "sol-ring"

        if outer_name not in card_name_to_id:
            skipped_unknown += 1
            continue

        outer_id = card_name_to_id[outer_name]

        with json_file.open("r") as f:
            data = json.load(f)

        pairs = extract_card_pairs(data)

        for pair in pairs:
            inner_name = pair["sanitized"]

            if inner_name not in card_name_to_id:
                # Card might exist in EDHREC but not in our seen_cards universe
                skipped_unknown += 1
                continue

            inner_id = card_name_to_id[inner_name]

            synergy_matrix[outer_id, inner_id] = pair["synergy"]
            lift_matrix[outer_id, inner_id] = pair["lift"]
            inclusion_matrix[outer_id, inner_id] = pair["inclusion"]
            potential_decks_matrix[outer_id, inner_id] = pair["potential_decks"]
            total_pairs += 1

    print(f"\nTotal pairs recorded: {total_pairs}")
    print(f"Skipped (unknown cards): {skipped_unknown}")

    # Convert to CSR format for efficient operations
    synergy_matrix = synergy_matrix.tocsr()
    lift_matrix = lift_matrix.tocsr()
    inclusion_matrix = inclusion_matrix.tocsr()
    potential_decks_matrix = potential_decks_matrix.tocsr()

    # Print raw matrix stats
    print(f"\n--- Raw Matrices ---")
    print(f"Synergy: {synergy_matrix.shape}, nnz={synergy_matrix.nnz}")
    print(f"Lift: {lift_matrix.shape}, nnz={lift_matrix.nnz}")
    print(f"Inclusion: {inclusion_matrix.shape}, nnz={inclusion_matrix.nnz}")
    print(f"Potential Decks: {potential_decks_matrix.shape}, nnz={potential_decks_matrix.nnz}")

    # Symmetrize the synergy matrix: S' = (S + S^T) / 2
    # For entries where both S[i,j] and S[j,i] exist, average them.
    # For entries where only one direction exists, keep that value.
    print("\nSymmetrizing synergy matrix...")
    synergy_t = synergy_matrix.T.tocsr()

    # Masks: where each direction has data
    has_forward = (synergy_matrix != 0)
    has_backward = (synergy_t != 0)
    has_both = has_forward.multiply(has_backward)

    # Average where both exist
    sym_both = has_both.multiply(synergy_matrix + synergy_t) / 2.0

    # Keep single direction where only one exists
    only_forward = has_forward - has_both
    only_backward = has_backward - has_both
    sym_forward = only_forward.multiply(synergy_matrix)
    sym_backward = only_backward.multiply(synergy_t)

    synergy_symmetric = (sym_both + sym_forward + sym_backward).tocsr()

    both_count = has_both.nnz // 2  # each pair counted twice
    only_fwd = only_forward.nnz
    only_bwd = only_backward.nnz
    print(f"  Pairs with both directions: {both_count}")
    print(f"  Pairs with only forward: {only_fwd}")
    print(f"  Pairs with only backward: {only_bwd}")
    print(f"  Symmetric matrix nnz: {synergy_symmetric.nnz}")

    # Build incoming synergy matrix: ReLU'd version (zero out negatives)
    print("\nBuilding incoming synergy matrix (ReLU'd)...")
    incoming_synergy_matrix = synergy_matrix.copy()
    incoming_synergy_matrix.data = np.maximum(incoming_synergy_matrix.data, 0)
    incoming_synergy_matrix.eliminate_zeros()
    print(f"  Incoming synergy matrix nnz: {incoming_synergy_matrix.nnz}")
    if incoming_synergy_matrix.nnz > 0:
        inc_data = incoming_synergy_matrix.data
        print(f"  Incoming synergy range: [{inc_data.min():.3f}, {inc_data.max():.3f}]")
        print(f"  Incoming synergy mean (non-zero): {inc_data.mean():.3f}")

    # Build incoming lift matrix: max(0, lift - 1)
    # Lift = 1.0 means "no more than random chance", so only excess matters.
    print("\nBuilding incoming lift matrix (max(0, lift - 1))...")
    incoming_lift_matrix = lift_matrix.copy()
    incoming_lift_matrix.data = np.maximum(incoming_lift_matrix.data - 1.0, 0)
    incoming_lift_matrix.eliminate_zeros()
    print(f"  Incoming lift matrix nnz: {incoming_lift_matrix.nnz}")
    if incoming_lift_matrix.nnz > 0:
        lift_inc_data = incoming_lift_matrix.data
        print(f"  Incoming lift range: [{lift_inc_data.min():.3f}, {lift_inc_data.max():.3f}]")
        print(f"  Incoming lift mean (non-zero): {lift_inc_data.mean():.3f}")

    # Build PPMI matrix: max(0, log(lift))
    # PPMI = Positive Pointwise Mutual Information
    # log(lift) normalizes the metric: compresses outliers (100 → 4.6),
    # preserves genuine synergy signal, information-theoretic foundation.
    print("\nBuilding PPMI matrix (max(0, log(lift)))...")
    incoming_ppmi_matrix = lift_matrix.copy()
    positive_mask = incoming_ppmi_matrix.data > 0
    incoming_ppmi_matrix.data[positive_mask] = np.log(incoming_ppmi_matrix.data[positive_mask])
    incoming_ppmi_matrix.data[~positive_mask] = 0
    incoming_ppmi_matrix.data = np.maximum(incoming_ppmi_matrix.data, 0)
    incoming_ppmi_matrix.eliminate_zeros()
    print(f"  PPMI matrix nnz: {incoming_ppmi_matrix.nnz}")
    if incoming_ppmi_matrix.nnz > 0:
        ppmi_data = incoming_ppmi_matrix.data
        print(f"  PPMI range: [{ppmi_data.min():.3f}, {ppmi_data.max():.3f}]")
        print(f"  PPMI mean (non-zero): {ppmi_data.mean():.3f}")
        print(f"  PPMI median (non-zero): {np.median(ppmi_data):.3f}")

    # Save all matrices
    matrices = {
        "synergy_matrix.pkl": synergy_matrix,
        "synergy_symmetric_matrix.pkl": synergy_symmetric,
        "incoming_synergy_matrix.pkl": incoming_synergy_matrix,
        "incoming_lift_matrix.pkl": incoming_lift_matrix,
        "incoming_ppmi_matrix.pkl": incoming_ppmi_matrix,
        "lift_matrix.pkl": lift_matrix,
        "inclusion_matrix.pkl": inclusion_matrix,
        "potential_decks_matrix.pkl": potential_decks_matrix,
    }

    for filename, matrix in matrices.items():
        with open(filename, "wb") as f:
            pickle.dump(matrix, f)
        print(f"Saved {filename}")

    # Print summary stats
    print(f"\n--- Summary ---")
    if synergy_matrix.nnz > 0:
        syn_data = synergy_matrix.data
        print(f"Synergy range: [{syn_data.min():.3f}, {syn_data.max():.3f}]")
        print(f"Synergy mean (non-zero): {syn_data.mean():.3f}")
        print(f"Synergy median (non-zero): {np.median(syn_data):.3f}")

    if synergy_symmetric.nnz > 0:
        sym_data = synergy_symmetric.data
        print(f"Symmetric synergy range: [{sym_data.min():.3f}, {sym_data.max():.3f}]")
        print(f"Symmetric synergy mean (non-zero): {sym_data.mean():.3f}")

    # Density
    density = synergy_matrix.nnz / (size * size) * 100
    print(f"Matrix density: {density:.4f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build sparse matrices from EDHREC JSON data")
    parser.add_argument(
        "--pool-size", type=int, default=None,
        help="Limit to top N cards by frequency (default: all cards)"
    )
    args = parser.parse_args()
    build_matrices(pool_size=args.pool_size)
