"""
Tests for synergy.py and incoming synergy matrix building.

Tests:
- ReLU application (zeroing out negatives)
- Pool scoping (N x N matrix for top N cards)
- Loading API consistency
- Column sum correctness for incoming synergy
"""

import pickle
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from load_cards_files import build_card_index
from synergy import (
    load_incoming_synergy,
    incoming_synergy_for,
    get_incoming_synergy_vector,
)


def test_relu_zeros_negatives():
    """Test that ReLU correctly zeroes out negative synergy values."""
    # Create a matrix with some negative values
    n = 5
    data = np.array([
        [0.0, 0.5, -0.3, 0.2, 0.0],
        [0.5, 0.0, 0.4, -0.1, 0.3],
        [-0.3, 0.4, 0.0, 0.6, -0.2],
        [0.2, -0.1, 0.6, 0.0, 0.1],
        [0.0, 0.3, -0.2, 0.1, 0.0],
    ])
    matrix = sp.csr_matrix(data)

    # Apply ReLU manually (as done in build_matrices)
    incoming = matrix.copy()
    incoming.data = np.maximum(incoming.data, 0)
    incoming.eliminate_zeros()

    # Check that negatives are zeroed
    assert incoming.min() >= 0, "ReLU should zero out all negatives"
    assert incoming[0, 2] == 0, "Negative value should be zeroed"
    assert incoming[2, 0] == 0, "Negative value should be zeroed"
    assert incoming[1, 3] == 0, "Negative value should be zeroed"
    assert incoming[3, 1] == 0, "Negative value should be zeroed"
    assert incoming[2, 4] == 0, "Negative value should be zeroed"
    assert incoming[4, 2] == 0, "Negative value should be zeroed"

    # Check that positives are preserved
    assert incoming[0, 1] == 0.5, "Positive value should be preserved"
    assert incoming[1, 0] == 0.5, "Positive value should be preserved"
    assert incoming[2, 3] == 0.6, "Positive value should be preserved"
    assert incoming[3, 2] == 0.6, "Positive value should be preserved"

    # Check that zeros remain zeros
    assert incoming[0, 0] == 0, "Diagonal zeros should remain"
    assert incoming[0, 4] == 0, "Zero entries should remain"


def test_pool_scoping():
    """Test that build_card_index with pool_size limits to top N cards."""
    # This test requires seen_cards.txt to exist
    # We'll test the function directly
    card_names_all, name_to_id_all = build_card_index(pool_size=None)
    card_names_10, name_to_id_10 = build_card_index(pool_size=10)
    card_names_5, name_to_id_5 = build_card_index(pool_size=5)

    # Check that pool_size limits the size
    assert len(card_names_10) == 10, "pool_size=10 should return 10 cards"
    assert len(card_names_5) == 5, "pool_size=5 should return 5 cards"
    assert len(card_names_all) >= len(card_names_10), "All cards should be >= pool subset"

    # Check that the pool is a prefix of the full list
    assert card_names_10 == card_names_all[:10], "Pool should be prefix of full list"
    assert card_names_5 == card_names_all[:5], "Pool should be prefix of full list"

    # Check that indices are consistent
    for i, name in enumerate(card_names_10):
        assert name_to_id_10[name] == i, "Indices should be sequential"
        assert name_to_id_all[name] == i, "Pool indices should match full indices"


def test_load_incoming_synergy_api(tmp_path, monkeypatch):
    """Test that load_incoming_synergy returns consistent shapes and mappings."""
    # Create temporary pickle files
    n = 10
    matrix = sp.csr_matrix(np.random.rand(n, n))
    card_names = [f"card-{i}" for i in range(n)]
    name_to_id = {name: i for i, name in enumerate(card_names)}

    # Save files to pytest tmp_path directory
    matrix_path = tmp_path / "incoming_synergy_matrix.pkl"
    names_path = tmp_path / "card_names.pkl"
    id_path = tmp_path / "card_name_to_id.pkl"

    with open(matrix_path, "wb") as f:
        pickle.dump(matrix, f)
    with open(names_path, "wb") as f:
        pickle.dump(card_names, f)
    with open(id_path, "wb") as f:
        pickle.dump(name_to_id, f)

    # Monkeypatch Path to point to temp directory
    import synergy
    original_matrix = synergy.INCOMING_SYNERGY_MATRIX
    original_names = synergy.CARD_NAMES_FILE
    original_id = synergy.CARD_NAME_TO_ID_FILE

    synergy.INCOMING_SYNERGY_MATRIX = matrix_path
    synergy.CARD_NAMES_FILE = names_path
    synergy.CARD_NAME_TO_ID_FILE = id_path

    try:
        # Load and verify
        loaded_matrix, loaded_names, loaded_id = load_incoming_synergy()

        assert loaded_matrix.shape == (n, n), "Matrix shape should match"
        assert len(loaded_names) == n, "Card names length should match"
        assert len(loaded_id) == n, "Name to ID mapping length should match"

        # Check consistency
        assert loaded_matrix.shape[0] == len(loaded_names), "Matrix rows should match names"
        assert loaded_matrix.shape[1] == len(loaded_names), "Matrix cols should match names"
        assert len(loaded_id) == len(loaded_names), "Mapping size should match names"

        # Check that mappings are correct
        for i, name in enumerate(loaded_names):
            assert loaded_id[name] == i, "Mapping should be consistent"
    finally:
        # Restore original paths
        synergy.INCOMING_SYNERGY_MATRIX = original_matrix
        synergy.CARD_NAMES_FILE = original_names
        synergy.CARD_NAME_TO_ID_FILE = original_id


def test_load_incoming_synergy_missing_files():
    """Test that load_incoming_synergy raises FileNotFoundError for missing files."""
    import synergy
    original_matrix = synergy.INCOMING_SYNERGY_MATRIX
    original_names = synergy.CARD_NAMES_FILE
    original_id = synergy.CARD_NAME_TO_ID_FILE

    # Point to non-existent files
    synergy.INCOMING_SYNERGY_MATRIX = Path("/nonexistent/matrix.pkl")
    synergy.CARD_NAMES_FILE = Path("/nonexistent/names.pkl")
    synergy.CARD_NAME_TO_ID_FILE = Path("/nonexistent/id.pkl")

    try:
        with pytest.raises(FileNotFoundError):
            load_incoming_synergy()
    finally:
        synergy.INCOMING_SYNERGY_MATRIX = original_matrix
        synergy.CARD_NAMES_FILE = original_names
        synergy.CARD_NAME_TO_ID_FILE = original_id


def test_incoming_synergy_for():
    """Test incoming_synergy_for computes column sum correctly."""
    n = 5
    # Create a matrix where we know the column sums
    data = np.array([
        [0.0, 0.5, 0.3, 0.2, 0.0],  # Column 0 sum = 0.0
        [0.5, 0.0, 0.4, 0.1, 0.3],  # Column 1 sum = 1.3
        [0.3, 0.4, 0.0, 0.6, 0.2],  # Column 2 sum = 1.5
        [0.2, 0.1, 0.6, 0.0, 0.1],  # Column 3 sum = 1.0
        [0.0, 0.3, 0.2, 0.1, 0.0],  # Column 4 sum = 0.6
    ])
    matrix = sp.csr_matrix(data)

    card_names = [f"card-{i}" for i in range(n)]
    name_to_id = {name: i for i, name in enumerate(card_names)}

    # Test each column
    assert incoming_synergy_for("card-0", matrix, card_names, name_to_id) == pytest.approx(0.0)
    assert incoming_synergy_for("card-1", matrix, card_names, name_to_id) == pytest.approx(1.3)
    assert incoming_synergy_for("card-2", matrix, card_names, name_to_id) == pytest.approx(1.5)
    assert incoming_synergy_for("card-3", matrix, card_names, name_to_id) == pytest.approx(1.0)
    assert incoming_synergy_for("card-4", matrix, card_names, name_to_id) == pytest.approx(0.6)

    # Test with non-normalized name (should still work via format_card_name)
    assert incoming_synergy_for("Card-1", matrix, card_names, name_to_id) == pytest.approx(1.3)

    # Test KeyError for unknown card
    with pytest.raises(KeyError):
        incoming_synergy_for("unknown-card", matrix, card_names, name_to_id)


def test_get_incoming_synergy_vector():
    """Test get_incoming_synergy_vector returns column sums."""
    n = 5
    data = np.array([
        [0.0, 0.5, 0.3, 0.2, 0.0],
        [0.5, 0.0, 0.4, 0.1, 0.3],
        [0.3, 0.4, 0.0, 0.6, 0.2],
        [0.2, 0.1, 0.6, 0.0, 0.1],
        [0.0, 0.3, 0.2, 0.1, 0.0],
    ])
    matrix = sp.csr_matrix(data)

    vector = get_incoming_synergy_vector(matrix)

    # Vector should be column vector (n x 1)
    assert vector.shape == (n, 1), "Vector should be column vector"

    # Check values match column sums
    expected_sums = [0.0, 1.3, 1.5, 1.0, 0.6]
    for i in range(n):
        assert vector[i, 0] == pytest.approx(expected_sums[i]), \
            f"Column {i} sum should match"


def test_incoming_synergy_with_negative_values():
    """Test that incoming synergy correctly handles negative values (should be zeroed)."""
    n = 4
    # Matrix with negatives that should be zeroed by ReLU
    data = np.array([
        [0.0, 0.5, -0.3, 0.2],
        [0.5, 0.0, 0.4, -0.1],
        [-0.3, 0.4, 0.0, 0.6],
        [0.2, -0.1, 0.6, 0.0],
    ])
    matrix = sp.csr_matrix(data)

    # Apply ReLU
    incoming = matrix.copy()
    incoming.data = np.maximum(incoming.data, 0)
    incoming.eliminate_zeros()

    # Column sums after ReLU
    # Column 0: 0.5 + 0.2 = 0.7 (negatives zeroed)
    # Column 1: 0.5 + 0.4 = 0.9 (negatives zeroed)
    # Column 2: 0.4 + 0.6 = 1.0 (negatives zeroed)
    # Column 3: 0.2 + 0.6 = 0.8 (negatives zeroed)

    card_names = [f"card-{i}" for i in range(n)]
    name_to_id = {name: i for i, name in enumerate(card_names)}

    assert incoming_synergy_for("card-0", incoming, card_names, name_to_id) == pytest.approx(0.7)
    assert incoming_synergy_for("card-1", incoming, card_names, name_to_id) == pytest.approx(0.9)
    assert incoming_synergy_for("card-2", incoming, card_names, name_to_id) == pytest.approx(1.0)
    assert incoming_synergy_for("card-3", incoming, card_names, name_to_id) == pytest.approx(0.8)
