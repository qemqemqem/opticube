"""
Tests for optimizer.py — the core optimization module.

All tests use synthetic data constructed in conftest.py fixtures.
No real card data or pickle files needed.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from optimizer import (
    SolverResult,
    symmetrize_matrix,
    reduce_matrix,
    _compute_synergy_score,
    find_good_subset_ilp,
    find_good_subset_greedy,
    spectral_analysis,
    spectral_prefilter,
    optimize,
)


# ===================================================================
# symmetrize_matrix
# ===================================================================

class TestSymmetrizeMatrix:
    """Verify the asymmetry fix: average both-direction, keep one-direction."""

    def test_already_symmetric_unchanged(self, symmetric_synergy_10):
        result = symmetrize_matrix(symmetric_synergy_10)
        diff = abs(result - result.T)
        assert diff.nnz == 0 or diff.max() < 1e-10

    def test_averages_both_directions(self, asymmetric_synergy_10):
        result = symmetrize_matrix(asymmetric_synergy_10)
        # (0,1)=4.0, (1,0)=6.0 -> average = 5.0
        assert abs(result[0, 1] - 5.0) < 1e-10
        assert abs(result[1, 0] - 5.0) < 1e-10

    def test_averages_second_pair(self, asymmetric_synergy_10):
        result = symmetrize_matrix(asymmetric_synergy_10)
        # (3,4)=2.0, (4,3)=8.0 -> average = 5.0
        assert abs(result[3, 4] - 5.0) < 1e-10
        assert abs(result[4, 3] - 5.0) < 1e-10

    def test_keeps_one_direction_forward(self, asymmetric_synergy_10):
        result = symmetrize_matrix(asymmetric_synergy_10)
        # (2,3)=3.0, (3,2)=0.0 -> should be 3.0 in both directions
        assert abs(result[2, 3] - 3.0) < 1e-10
        assert abs(result[3, 2] - 3.0) < 1e-10

    def test_keeps_one_direction_backward(self, asymmetric_synergy_10):
        result = symmetrize_matrix(asymmetric_synergy_10)
        # (8,7)=2.0, (7,8)=0.0 -> should be 2.0 in both directions
        assert abs(result[7, 8] - 2.0) < 1e-10
        assert abs(result[8, 7] - 2.0) < 1e-10

    def test_result_is_symmetric(self, asymmetric_synergy_10):
        result = symmetrize_matrix(asymmetric_synergy_10)
        diff = abs(result - result.T)
        if diff.nnz > 0:
            assert diff.max() < 1e-10

    def test_preserves_zeros(self, asymmetric_synergy_10):
        result = symmetrize_matrix(asymmetric_synergy_10)
        # (0,5) and (5,0) are both zero in input -> should stay zero
        assert result[0, 5] == 0.0
        assert result[5, 0] == 0.0

    def test_output_is_sparse(self, asymmetric_synergy_10):
        result = symmetrize_matrix(asymmetric_synergy_10)
        assert sp.issparse(result)


# ===================================================================
# reduce_matrix
# ===================================================================

class TestReduceMatrix:
    """Verify the index mapping fix in pre-filtering."""

    def test_returns_correct_shape(self, symmetric_synergy_10):
        reduced, top_indices = reduce_matrix(symmetric_synergy_10, top_n=5)
        assert reduced.shape == (5, 5)
        assert len(top_indices) == 5

    def test_top_indices_map_back_correctly(self, symmetric_synergy_10):
        """Values in reduced matrix should match original at mapped positions."""
        reduced, top_indices = reduce_matrix(symmetric_synergy_10, top_n=5)
        reduced_dense = reduced.toarray() if sp.issparse(reduced) else reduced
        orig_dense = symmetric_synergy_10.toarray()

        for ri in range(5):
            for rj in range(5):
                oi, oj = top_indices[ri], top_indices[rj]
                assert abs(reduced_dense[ri, rj] - orig_dense[oi, oj]) < 1e-10

    def test_selects_highest_synergy_cards(self, symmetric_synergy_10):
        """Block A (cards 0-4) has higher total synergy; they should be selected."""
        reduced, top_indices = reduce_matrix(symmetric_synergy_10, top_n=5)
        # Cards 0-4 each have row+col sum of 4*5.0 + 5*0.5 + 4*5.0 + 5*0.5 = 45.0
        # Cards 5-9 each have row+col sum of 4*3.0 + 5*0.5 + 4*3.0 + 5*0.5 = 29.0
        # So top-5 should be cards 0-4
        assert set(top_indices) == {0, 1, 2, 3, 4}

    def test_full_size_returns_everything(self, symmetric_synergy_10):
        """If top_n == n, get back the full matrix."""
        n = symmetric_synergy_10.shape[0]
        reduced, top_indices = reduce_matrix(symmetric_synergy_10, top_n=n)
        assert reduced.shape == (n, n)
        assert len(top_indices) == n

    def test_reduce_single(self, symmetric_synergy_10):
        """Reducing to 1 card should return a 1x1 matrix."""
        reduced, top_indices = reduce_matrix(symmetric_synergy_10, top_n=1)
        assert reduced.shape == (1, 1)
        assert len(top_indices) == 1


# ===================================================================
# _compute_synergy_score
# ===================================================================

class TestComputeSynergyScore:
    """Verify pairwise synergy calculation on upper triangle."""

    def test_known_score(self, synergy_with_clear_optimum):
        # Cards 0,1,2: pairwise synergy 10.0 each, 3 pairs -> 30.0
        score = _compute_synergy_score(synergy_with_clear_optimum, [0, 1, 2])
        assert abs(score - 30.0) < 1e-10

    def test_weak_group_score(self, synergy_with_clear_optimum):
        # Cards 3,4,5: pairwise synergy 1.0 each, 3 pairs -> 3.0
        score = _compute_synergy_score(synergy_with_clear_optimum, [3, 4, 5])
        assert abs(score - 3.0) < 1e-10

    def test_empty_set(self, synergy_with_clear_optimum):
        assert _compute_synergy_score(synergy_with_clear_optimum, []) == 0.0

    def test_single_card(self, synergy_with_clear_optimum):
        assert _compute_synergy_score(synergy_with_clear_optimum, [0]) == 0.0

    def test_cross_group(self, synergy_with_clear_optimum):
        # Cards 0 and 3 have zero cross-group synergy
        score = _compute_synergy_score(synergy_with_clear_optimum, [0, 3])
        assert abs(score - 0.0) < 1e-10

    def test_full_matrix(self, symmetric_synergy_10):
        """Score of all 10 cards: sum of all upper-triangle entries."""
        all_indices = list(range(10))
        score = _compute_synergy_score(symmetric_synergy_10, all_indices)
        dense = symmetric_synergy_10.toarray()
        expected = np.triu(dense, k=1).sum()
        assert abs(score - expected) < 1e-10


# ===================================================================
# find_good_subset_ilp
# ===================================================================

class TestFindGoodSubsetILP:
    """ILP solver on small synthetic problems with known optima."""

    def test_finds_optimal_3_from_6(self, synergy_with_clear_optimum):
        """Must pick {0,1,2} from the 6-card matrix."""
        result = find_good_subset_ilp(
            synergy_with_clear_optimum, set_size=3, time_limit=30,
        )
        assert isinstance(result, SolverResult)
        assert set(result.indices) == {0, 1, 2}

    def test_correct_set_size(self, symmetric_synergy_10):
        result = find_good_subset_ilp(
            symmetric_synergy_10, set_size=5, time_limit=30,
        )
        assert len(result.indices) == 5

    def test_picks_strong_block(self, symmetric_synergy_10):
        """With set_size=5, should pick the block A (cards 0-4) with synergy=5.0."""
        result = find_good_subset_ilp(
            symmetric_synergy_10, set_size=5, time_limit=30,
        )
        assert set(result.indices) == {0, 1, 2, 3, 4}

    def test_set_size_1(self, symmetric_synergy_10):
        """Selecting 1 card should return exactly 1 index."""
        result = find_good_subset_ilp(
            symmetric_synergy_10, set_size=1, time_limit=10,
        )
        assert len(result.indices) == 1

    def test_all_zero_matrix(self):
        """If all synergies are zero, any selection is equally good."""
        zero = sp.csr_matrix((5, 5))
        result = find_good_subset_ilp(zero, set_size=3, time_limit=10)
        assert len(result.indices) == 3
        assert len(set(result.indices)) == 3  # all distinct

    def test_solver_result_has_quality_metrics(self, synergy_with_clear_optimum):
        """SolverResult should include gap, wall_time, status."""
        result = find_good_subset_ilp(
            synergy_with_clear_optimum, set_size=3, time_limit=30,
        )
        assert result.method == 'cbc'
        assert result.status == 'optimal'
        assert result.wall_time > 0
        assert result.gap is not None
        assert result.score >= 0

    def test_highs_solver(self, synergy_with_clear_optimum):
        """HiGHS backend should find the same optimum."""
        try:
            from pulp import HiGHS
            if not HiGHS(msg=0).available():
                pytest.skip("HiGHS solver not available")
        except ImportError:
            pytest.skip("HiGHS not available")

        result = find_good_subset_ilp(
            synergy_with_clear_optimum, set_size=3, time_limit=30,
            solver='highs',
        )
        assert set(result.indices) == {0, 1, 2}
        assert result.method == 'highs'


# ===================================================================
# find_good_subset_greedy
# ===================================================================

class TestFindGoodSubsetGreedy:
    """Greedy + local search on synthetic data."""

    def test_finds_strong_group(self, synergy_with_clear_optimum):
        """Greedy should find {0,1,2} or at least a high-scoring set."""
        result = find_good_subset_greedy(
            synergy_with_clear_optimum, set_size=3,
            num_restarts=5, max_no_improve=200, seed=42,
        )
        assert isinstance(result, SolverResult)
        assert len(result.indices) == 3
        # The optimal score is 30.0; greedy should find it on this tiny problem
        assert result.score >= 29.0

    def test_correct_set_size(self, symmetric_synergy_10):
        result = find_good_subset_greedy(
            symmetric_synergy_10, set_size=5,
            num_restarts=3, max_no_improve=100, seed=42,
        )
        assert len(result.indices) == 5

    def test_score_positive(self, symmetric_synergy_10):
        result = find_good_subset_greedy(
            symmetric_synergy_10, set_size=5,
            num_restarts=3, max_no_improve=100, seed=42,
        )
        assert result.score > 0

    def test_deterministic_with_seed(self, synergy_with_clear_optimum):
        """Same seed -> same result."""
        r1 = find_good_subset_greedy(
            synergy_with_clear_optimum, set_size=3,
            num_restarts=3, max_no_improve=100, seed=123,
        )
        r2 = find_good_subset_greedy(
            synergy_with_clear_optimum, set_size=3,
            num_restarts=3, max_no_improve=100, seed=123,
        )
        assert set(r1.indices) == set(r2.indices)
        assert abs(r1.score - r2.score) < 1e-10

    def test_larger_problem(self, synergy_50):
        """Greedy should handle 50 cards comfortably."""
        result = find_good_subset_greedy(
            synergy_50, set_size=10,
            num_restarts=3, max_no_improve=200, seed=42,
        )
        assert len(result.indices) == 10
        assert result.score > 0

    def test_greedy_beats_random_baseline(self, synergy_50):
        """Greedy should beat a random selection on average."""
        result = find_good_subset_greedy(
            synergy_50, set_size=10,
            num_restarts=3, max_no_improve=200, seed=42,
        )

        rng = np.random.RandomState(99)
        random_scores = []
        for _ in range(50):
            rand_idx = rng.choice(50, size=10, replace=False)
            random_scores.append(_compute_synergy_score(synergy_50, rand_idx))

        avg_random = np.mean(random_scores)
        assert result.score > avg_random

    def test_greedy_reports_heuristic_status(self, synergy_with_clear_optimum):
        """Greedy result should have status='heuristic' and gap=None."""
        result = find_good_subset_greedy(
            synergy_with_clear_optimum, set_size=3,
            num_restarts=2, max_no_improve=50, seed=42,
        )
        assert result.method == 'greedy'
        assert result.status == 'heuristic'
        assert result.gap is None
        assert result.best_bound is None
        assert result.wall_time > 0
        assert result.iterations > 0


# ===================================================================
# spectral_analysis
# ===================================================================

class TestSpectralAnalysis:
    """Spectral decomposition on block-structured matrices."""

    def test_returns_eigenvalues_and_vectors(self, synergy_50):
        eigenvalues, eigenvectors = spectral_analysis(synergy_50, n_components=5)
        assert len(eigenvalues) == 5
        assert eigenvectors.shape == (50, 5)

    def test_eigenvalues_descending(self, synergy_50):
        eigenvalues, _ = spectral_analysis(synergy_50, n_components=5)
        for i in range(len(eigenvalues) - 1):
            assert eigenvalues[i] >= eigenvalues[i + 1] - 1e-10

    def test_dominant_eigenvalue_positive(self, synergy_50):
        eigenvalues, _ = spectral_analysis(synergy_50, n_components=3)
        assert eigenvalues[0] > 0

    def test_n_components_clamped(self, synergy_with_clear_optimum):
        """If matrix is 6x6, n_components should be clamped to 4."""
        eigenvalues, eigenvectors = spectral_analysis(
            synergy_with_clear_optimum, n_components=10,
        )
        assert len(eigenvalues) == 4  # min(10, 6-2)
        assert eigenvectors.shape == (6, 4)


# ===================================================================
# spectral_prefilter
# ===================================================================

class TestSpectralPrefilter:
    """Diversity-based candidate selection via farthest-point sampling."""

    def test_returns_correct_count(self, synergy_50):
        selected = spectral_prefilter(synergy_50, target_n=20, n_components=5)
        assert len(selected) == 20

    def test_all_unique(self, synergy_50):
        selected = spectral_prefilter(synergy_50, target_n=20, n_components=5)
        assert len(set(selected)) == 20

    def test_valid_indices(self, synergy_50):
        selected = spectral_prefilter(synergy_50, target_n=20, n_components=5)
        for idx in selected:
            assert 0 <= idx < 50

    def test_full_size_returns_all(self, synergy_50):
        """If target_n >= n, return all indices."""
        selected = spectral_prefilter(synergy_50, target_n=100, n_components=5)
        assert len(selected) == 50

    def test_diversity_across_clusters(self, synergy_50):
        """With 5 clusters, selecting 10 cards should hit multiple clusters."""
        selected = spectral_prefilter(synergy_50, target_n=10, n_components=5)
        clusters_hit = set()
        for idx in selected:
            clusters_hit.add(idx // 10)  # cluster = floor(idx/10)
        # Should hit at least 3 of the 5 clusters
        assert len(clusters_hit) >= 3


# ===================================================================
# optimize (full pipeline)
# ===================================================================

class TestOptimizePipeline:
    """End-to-end tests for the optimize() pipeline."""

    def test_ilp_pipeline(self, asymmetric_synergy_10, card_names_10):
        """Full pipeline: symmetrize -> reduce -> ILP -> map back to names."""
        names, indices, score = optimize(
            synergy_matrix=asymmetric_synergy_10,
            card_names=card_names_10,
            set_size=3,
            method='ilp',
            prefilter='topn',
            prefilter_multiplier=50,
            time_limit=30,
        )
        assert len(names) == 3
        assert len(indices) == 3
        assert all(isinstance(n, str) for n in names)
        assert all(0 <= i < 10 for i in indices)

    def test_greedy_pipeline(self, symmetric_synergy_10, card_names_10):
        names, indices, score = optimize(
            synergy_matrix=symmetric_synergy_10,
            card_names=card_names_10,
            set_size=5,
            method='greedy',
            prefilter='topn',
            prefilter_multiplier=50,
            time_limit=30,
            num_restarts=3,
        )
        assert len(names) == 5
        assert score > 0

    def test_names_match_indices(self, symmetric_synergy_10, card_names_10):
        """Returned names must correspond to returned indices."""
        names, indices, _ = optimize(
            synergy_matrix=symmetric_synergy_10,
            card_names=card_names_10,
            set_size=3,
            method='ilp',
            time_limit=30,
        )
        expected_names = sorted([card_names_10[i] for i in indices])
        assert sorted(names) == expected_names

    def test_invalid_method_raises(self, symmetric_synergy_10, card_names_10):
        with pytest.raises(ValueError, match="Unknown method"):
            optimize(
                synergy_matrix=symmetric_synergy_10,
                card_names=card_names_10,
                set_size=3,
                method='quantum',
            )

    def test_pipeline_with_clear_optimum(self, card_names_10):
        """
        Build a 10-card matrix where cards 0-2 are obviously best,
        and verify the pipeline finds them.
        """
        n = 10
        data = np.zeros((n, n))
        for i in range(3):
            for j in range(i + 1, 3):
                data[i, j] = 20.0
                data[j, i] = 20.0
        # Weak connections elsewhere
        for i in range(3, n):
            for j in range(i + 1, n):
                data[i, j] = 0.1
                data[j, i] = 0.1

        matrix = sp.csr_matrix(data)
        names, indices, score = optimize(
            synergy_matrix=matrix,
            card_names=card_names_10,
            set_size=3,
            method='ilp',
            time_limit=30,
        )
        assert set(indices) == {0, 1, 2}
        expected = {"Sol Ring", "Lightning Bolt", "Llanowar Elves"}
        assert set(names) == expected


# ===================================================================
# CP-SAT Solver
# ===================================================================

class TestFindGoodSubsetCPSAT:
    """CP-SAT solver on small synthetic problems with known optima."""

    def _cpsat_available(self):
        try:
            from ortools.sat.python import cp_model  # noqa: F401
            return True
        except ImportError:
            return False

    def test_finds_optimal_3_from_6(self, synergy_with_clear_optimum):
        if not self._cpsat_available():
            pytest.skip("ortools not installed")
        from optimizer import find_good_subset_cpsat
        result = find_good_subset_cpsat(
            synergy_with_clear_optimum, set_size=3, time_limit=30,
        )
        assert isinstance(result, SolverResult)
        assert set(result.indices) == {0, 1, 2}
        assert result.method == 'cpsat'

    def test_correct_set_size(self, symmetric_synergy_10):
        if not self._cpsat_available():
            pytest.skip("ortools not installed")
        from optimizer import find_good_subset_cpsat
        result = find_good_subset_cpsat(
            symmetric_synergy_10, set_size=5, time_limit=30,
        )
        assert len(result.indices) == 5

    def test_picks_strong_block(self, symmetric_synergy_10):
        if not self._cpsat_available():
            pytest.skip("ortools not installed")
        from optimizer import find_good_subset_cpsat
        result = find_good_subset_cpsat(
            symmetric_synergy_10, set_size=5, time_limit=30,
        )
        assert set(result.indices) == {0, 1, 2, 3, 4}

    def test_reports_gap_and_bound(self, synergy_with_clear_optimum):
        """CP-SAT should report an exact gap when it proves optimality."""
        if not self._cpsat_available():
            pytest.skip("ortools not installed")
        from optimizer import find_good_subset_cpsat
        result = find_good_subset_cpsat(
            synergy_with_clear_optimum, set_size=3, time_limit=30,
        )
        assert result.status == 'optimal'
        assert result.gap is not None
        assert result.gap < 0.01  # should be essentially 0
        assert result.best_bound is not None
        assert result.wall_time > 0

    def test_cpsat_pipeline(self, symmetric_synergy_10, card_names_10):
        """CP-SAT should work through the optimize() pipeline."""
        if not self._cpsat_available():
            pytest.skip("ortools not installed")
        names, indices, score = optimize(
            synergy_matrix=symmetric_synergy_10,
            card_names=card_names_10,
            set_size=5,
            method='ilp',
            solver='cpsat',
            time_limit=30,
        )
        assert len(names) == 5
        assert set(indices) == {0, 1, 2, 3, 4}
