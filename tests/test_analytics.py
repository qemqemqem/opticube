"""
Tests for analytics.py — matrix analysis and random sampling baseline.

Uses synthetic data from conftest.py.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from analytics import find_good_subset_random


# ===================================================================
# find_good_subset_random
# ===================================================================

class TestFindGoodSubsetRandom:
    """Random sampling baseline correctness and properties."""

    def test_returns_correct_size(self, symmetric_synergy_10, card_names_10):
        subset, goodness = find_good_subset_random(
            symmetric_synergy_10, card_names_10, set_size=5, num_tries=50,
        )
        assert len(subset) == 5

    def test_positive_goodness(self, symmetric_synergy_10, card_names_10):
        subset, goodness = find_good_subset_random(
            symmetric_synergy_10, card_names_10, set_size=5, num_tries=50,
        )
        assert goodness > 0

    def test_indices_in_range(self, symmetric_synergy_10, card_names_10):
        subset, _ = find_good_subset_random(
            symmetric_synergy_10, card_names_10, set_size=5, num_tries=50,
        )
        for idx in subset:
            assert 0 <= idx < 10

    def test_no_duplicates(self, symmetric_synergy_10, card_names_10):
        subset, _ = find_good_subset_random(
            symmetric_synergy_10, card_names_10, set_size=5, num_tries=50,
        )
        assert len(set(subset)) == 5

    def test_set_size_1(self, symmetric_synergy_10, card_names_10):
        subset, goodness = find_good_subset_random(
            symmetric_synergy_10, card_names_10, set_size=1, num_tries=10,
        )
        assert len(subset) == 1
        # Single card has zero pairwise synergy
        assert goodness == 0

    def test_more_tries_at_least_as_good(self, symmetric_synergy_10, card_names_10):
        """More tries should produce scores >= fewer tries (probabilistically)."""
        np.random.seed(42)
        _, score_few = find_good_subset_random(
            symmetric_synergy_10, card_names_10, set_size=5, num_tries=10,
        )
        np.random.seed(42)
        _, score_many = find_good_subset_random(
            symmetric_synergy_10, card_names_10, set_size=5, num_tries=500,
        )
        # With same seed start, more tries should explore at least as much
        assert score_many >= score_few
