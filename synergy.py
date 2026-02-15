"""
Synergy Matrix Loading API

Provides a clean interface for loading the incoming synergy matrix and
performing lookups. The incoming synergy matrix is a sparse square matrix
where M[b, a] = max(0, S[b, a]) -- the ReLU'd synergy contribution from
card b to card a. Column sums give the total incoming synergy per card.

Usage:
    from synergy import load_incoming_synergy, incoming_synergy_for

    matrix, card_names, name_to_id = load_incoming_synergy()
    total = incoming_synergy_for("sol-ring", matrix, card_names, name_to_id)
"""

import pickle
from pathlib import Path

import scipy.sparse as sp

from utils import format_card_name

INCOMING_SYNERGY_MATRIX = Path("incoming_synergy_matrix.pkl")
CARD_NAMES_FILE = Path("card_names.pkl")
CARD_NAME_TO_ID_FILE = Path("card_name_to_id.pkl")


def load_incoming_synergy() -> tuple[sp.csr_matrix, list[str], dict[str, int]]:
    """
    Load the incoming synergy matrix for the card pool.

    Returns:
        matrix: Sparse CSR matrix M where M[b, a] = max(0, S[b, a]).
                Column a's sum = total incoming synergy for card a.
        card_names: List mapping index -> card name slug.
        name_to_id: Dict mapping card name slug -> index.

    Raises:
        FileNotFoundError: If the required pickle files don't exist.
            Run `python load_cards_files.py` or `python run_pipeline.py` first.
    """
    if not INCOMING_SYNERGY_MATRIX.exists():
        raise FileNotFoundError(
            f"Incoming synergy matrix not found at {INCOMING_SYNERGY_MATRIX}. "
            "Run 'python load_cards_files.py' or 'python run_pipeline.py' first."
        )

    if not CARD_NAMES_FILE.exists():
        raise FileNotFoundError(
            f"Card names file not found at {CARD_NAMES_FILE}. "
            "Run 'python load_cards_files.py' or 'python run_pipeline.py' first."
        )

    if not CARD_NAME_TO_ID_FILE.exists():
        raise FileNotFoundError(
            f"Card name to ID mapping not found at {CARD_NAME_TO_ID_FILE}. "
            "Run 'python load_cards_files.py' or 'python run_pipeline.py' first."
        )

    with open(INCOMING_SYNERGY_MATRIX, "rb") as f:
        matrix = pickle.load(f)

    with open(CARD_NAMES_FILE, "rb") as f:
        card_names = pickle.load(f)

    with open(CARD_NAME_TO_ID_FILE, "rb") as f:
        name_to_id = pickle.load(f)

    # Verify consistency
    if matrix.shape[0] != len(card_names) or matrix.shape[1] != len(card_names):
        raise ValueError(
            f"Matrix shape {matrix.shape} doesn't match card_names length {len(card_names)}"
        )

    if len(name_to_id) != len(card_names):
        raise ValueError(
            f"name_to_id size {len(name_to_id)} doesn't match card_names length {len(card_names)}"
        )

    return matrix, card_names, name_to_id


def incoming_synergy_for(
    card_name: str,
    matrix: sp.csr_matrix,
    card_names: list[str],
    name_to_id: dict[str, int],
) -> float:
    """
    Compute total incoming synergy for a card (sum of its column).

    Args:
        card_name: Card name (will be normalized via format_card_name).
        matrix: Incoming synergy matrix from load_incoming_synergy().
        card_names: Card names list from load_incoming_synergy().
        name_to_id: Name to ID mapping from load_incoming_synergy().

    Returns:
        Total incoming synergy (sum of column for this card).

    Raises:
        KeyError: If the card name is not in the pool.
    """
    normalized = format_card_name(card_name)
    if normalized not in name_to_id:
        raise KeyError(f"Card '{card_name}' (normalized: '{normalized}') not found in pool")

    idx = name_to_id[normalized]
    col = matrix.getcol(idx)
    return float(col.sum())


def get_incoming_synergy_vector(
    matrix: sp.csr_matrix,
) -> sp.csr_matrix:
    """
    Get the vector of total incoming synergy per card (column sums).

    Args:
        matrix: Incoming synergy matrix from load_incoming_synergy().

    Returns:
        Sparse column vector where entry i is the total incoming synergy for card i.
    """
    return matrix.sum(axis=0).T
