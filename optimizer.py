"""
Opticube Optimizer

Finds optimal card subsets maximizing pairwise synergy,
subject to optional cube design constraints (color balance, mana curve, card types).

Supports multiple solving strategies:
- ILP via CBC, HiGHS, or OR-Tools CP-SAT - exact for small-to-medium problems
- Greedy + Local Search - scalable to k=360 from pools of 5000+ cards
- Spectral pre-filtering for diverse candidate selection

All solvers return SolverResult with quality metrics (gap, bound, wall time).

Usage:
    python optimizer.py --set-size 10 --method ilp --solver cpsat
    python optimizer.py --set-size 10 --method ilp --solver cbc --gap-rel 0.01
    python optimizer.py --set-size 50 --method greedy --num-restarts 10
    python optimizer.py --set-size 360 --method log-synergy --max-no-improve 5000
    python optimizer.py --spectral-clusters 8
"""

import argparse
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from tqdm import tqdm

console = Console(width=100, soft_wrap=True)


# ---------------------------------------------------------------------------
# Effort presets
# ---------------------------------------------------------------------------

EFFORT_PRESETS = {
    'quick':    {'num_restarts': 2,  'max_no_improve': 500},
    'normal':   {'num_restarts': 5,  'max_no_improve': 2000},
    'thorough': {'num_restarts': 20, 'max_no_improve': 5000},
    'max':      {'num_restarts': 50, 'max_no_improve': 10000},
}


def resolve_effort(effort, num_restarts=None, max_no_improve=None):
    """Resolve effort preset with optional explicit overrides.

    Returns (num_restarts, max_no_improve) tuple.
    Explicit values override the preset; None falls through to the preset default.
    """
    preset = EFFORT_PRESETS[effort]
    return (
        num_restarts if num_restarts is not None else preset['num_restarts'],
        max_no_improve if max_no_improve is not None else preset['max_no_improve'],
    )


# ---------------------------------------------------------------------------
# Solver result
# ---------------------------------------------------------------------------

@dataclass
class SolverResult:
    """Standardized result from any solver, including quality metrics.

    Attributes:
        indices: Selected card indices into the input matrix.
        score: Objective value achieved.
        method: Solver backend ('cbc', 'highs', 'cpsat', 'greedy').
        wall_time: Wall clock seconds.
        status: 'optimal' | 'feasible' | 'heuristic' | 'infeasible' | 'timeout'.
        gap: Relative optimality gap (0.0 = proven optimal). None for heuristics.
        best_bound: Upper bound on optimal objective. None if unavailable.
        iterations: Solver iterations or swap attempts.
    """
    indices: list
    score: float
    method: str
    wall_time: float
    status: str
    gap: Optional[float] = None
    best_bound: Optional[float] = None
    iterations: Optional[int] = None


def _print_solver_report(result, effort_name=None, metric=None, objective=None):
    """Display solver quality metrics to console."""
    from rich.panel import Panel

    lines = []
    lines.append(f"Method:     {result.method.upper()}")
    if metric:
        lines.append(f"Metric:     {metric}")
    if objective:
        obj_label = "ln(1 + incoming)" if objective == 'log' else "sum(incoming)"
        lines.append(f"Objective:  {obj_label}")
    if effort_name and result.status == 'heuristic':
        preset = EFFORT_PRESETS.get(effort_name, {})
        lines.append(f"Effort:     {effort_name} "
                     f"({preset.get('num_restarts', '?')} restarts, "
                     f"patience {preset.get('max_no_improve', '?')})")
    lines.append(f"Status:     {result.status}")
    lines.append(f"Score:      {result.score:.4f}")
    lines.append(f"Wall time:  {result.wall_time:.1f}s")

    if result.gap is not None:
        if result.gap < 1e-6:
            lines.append("Gap:        ~0% (proven optimal)")
        else:
            lines.append(f"Gap:        ≤ {result.gap * 100:.2f}%")
    if result.best_bound is not None:
        lines.append(f"Best bound: {result.best_bound:.4f}")
    if result.iterations is not None:
        lines.append(f"Iterations: {result.iterations:,}")
    lines.append(f"Cards:      {len(result.indices)} selected")

    # Effort-based tip for heuristic solvers
    if effort_name and result.status == 'heuristic':
        tips = {
            'quick': 'Tip: Quick run. Use --effort normal for better quality.',
            'normal': 'Tip: Try --effort thorough or --seed <N> to check stability.',
            'thorough': 'Tip: Try --seed <N> to verify stability.',
            'max': 'Tip: Maximum effort. Result should be near-optimal for this heuristic.',
        }
        tip = tips.get(effort_name)
        if tip:
            lines.append("")
            lines.append(tip)

    console.print()
    console.print(Panel("\n".join(lines), title="Solver Report", border_style="green"))


# ---------------------------------------------------------------------------
# Matrix preprocessing
# ---------------------------------------------------------------------------

def symmetrize_matrix(S):
    """
    Symmetrize an asymmetric synergy matrix.

    - Where both S[i,j] and S[j,i] exist (non-zero), average them.
    - Where only one direction exists, use it as-is.

    This fixes the asymmetry bug where the ILP arbitrarily picked whichever
    direction had i <= j, discarding the other.
    """
    S = S.tocsr()
    # Entries where both directions are non-zero
    S_both = S.multiply(S.T > 0)
    # Entries where only one direction is non-zero
    S_one = S.multiply(S.T == 0)
    S_sym = (S_both + S_both.T) / 2 + S_one + S_one.T
    return S_sym.tocsr()


def reduce_matrix(matrix, top_n):
    """
    Reduce the matrix to the top_n most promising candidates by total synergy.

    Returns:
        reduced_matrix: The submatrix of top candidates
        top_indices: Array mapping reduced indices -> original indices

    This fixes the index mapping bug where the original code discarded the
    mapping, causing card name lookups to return wrong names.
    """
    row_sums = np.array(matrix.sum(axis=1)).flatten()
    col_sums = np.array(matrix.sum(axis=0)).flatten()
    total_sums = row_sums + col_sums
    top_indices = np.argsort(total_sums)[-top_n:]
    reduced_matrix = matrix[np.ix_(top_indices, top_indices)]
    return reduced_matrix, top_indices


# ---------------------------------------------------------------------------
# ILP Solver (fixed)
# ---------------------------------------------------------------------------

def find_good_subset_ilp(matrix, set_size, card_properties=None,
                         time_limit=120, gap_rel=0.05, solver='cbc',
                         threads=0):
    """
    Find optimal subset using Integer Linear Programming.

    Supports CBC (default, bundled with PuLP) and HiGHS (requires highspy).
    Returns SolverResult with quality metrics including optimality gap.

    Args:
        matrix: Symmetrized sparse synergy matrix (n x n)
        set_size: Number of cards to select
        card_properties: Optional dict with per-card metadata for constraints
        time_limit: Solver time limit in seconds
        gap_rel: Relative optimality gap (0.05 = 5%). Set to 0 for exact optimum.
        solver: 'cbc' or 'highs'
        threads: Number of threads (HiGHS only, 0 = auto)

    Returns:
        SolverResult with selected indices and quality metrics
    """
    t0 = time.time()
    n = matrix.shape[0]

    console.print(Markdown(f"## ILP Solver ({solver.upper()})"))
    console.print(f"  Matrix: {n}x{n}, {matrix.nnz} non-zero entries")
    console.print(f"  Selecting {set_size} cards, time limit {time_limit}s, gap {gap_rel * 100}%")

    prob = LpProblem("MaxSynergy", LpMaximize)
    choices = LpVariable.dicts("x", range(n), cat=LpBinary)

    # Convert to COO for iteration over non-zero entries
    matrix_coo = matrix.tocoo()

    # Create product variables y_{ij} for linearization (only upper triangle)
    product_vars = {}
    for k in range(len(matrix_coo.data)):
        i, j = int(matrix_coo.row[k]), int(matrix_coo.col[k])
        if i <= j:
            product_vars[(i, j)] = LpVariable(f"y_{i}_{j}", cat=LpBinary)

    # Linearization constraints: y_ij = x_i * x_j
    for (i, j), var in product_vars.items():
        if i != j:
            prob += var <= choices[i]
            prob += var <= choices[j]
            prob += var >= choices[i] + choices[j] - 1
        else:
            prob += var == choices[i]

    # Objective: maximize total synergy (upper triangle only, no double-counting)
    prob += lpSum(
        matrix_coo.data[k] * product_vars[(int(matrix_coo.row[k]), int(matrix_coo.col[k]))]
        for k in range(len(matrix_coo.data))
        if (int(matrix_coo.row[k]), int(matrix_coo.col[k])) in product_vars
    )

    # Cardinality constraint
    prob += lpSum(choices[i] for i in range(n)) == set_size

    # Optional constraints from card metadata
    if card_properties is not None:
        _add_metadata_constraints(prob, choices, n, set_size, card_properties)

    # Select solver backend
    if solver == 'highs':
        try:
            from pulp import HiGHS
            pulp_solver = HiGHS(
                timeLimit=time_limit,
                gapRel=gap_rel,
                msg=1,
                threads=threads if threads > 0 else None,
            )
        except ImportError:
            console.print("  [yellow]HiGHS not available, falling back to CBC[/yellow]")
            solver = 'cbc'
            pulp_solver = PULP_CBC_CMD(timeLimit=time_limit, gapRel=gap_rel, msg=1)
    else:
        pulp_solver = PULP_CBC_CMD(timeLimit=time_limit, gapRel=gap_rel, msg=1)

    prob.solve(pulp_solver)
    wall_time = time.time() - t0

    solution_indices = [i for i in range(n) if choices[i].varValue == 1]
    obj_val = prob.objective.value() if prob.objective.value() is not None else 0

    # PuLP status 1 = "Optimal" (within gap_rel tolerance)
    if prob.status == 1:
        status_str = 'optimal'
        gap = gap_rel  # guaranteed within this tolerance
    elif solution_indices:
        status_str = 'feasible'
        gap = None  # timed out, gap unknown
    else:
        status_str = 'infeasible'
        gap = None

    result = SolverResult(
        indices=solution_indices,
        score=float(obj_val),
        method=solver,
        wall_time=wall_time,
        status=status_str,
        gap=gap,
        best_bound=None,  # PuLP doesn't expose the dual bound
    )
    _print_solver_report(result)
    return result


def _add_metadata_constraints(prob, choices, n, set_size, props):
    """
    Add cube design constraints to the ILP using card metadata.

    Constraints are scaled proportionally to set_size (relative to a 360-card cube).
    Only applied when set_size is large enough for them to make sense.
    """
    scale = set_size / 360.0

    # Color balance constraints (only for set_size >= 30)
    if 'color_identity' in props and set_size >= 30:
        for color in ['W', 'U', 'B', 'R', 'G']:
            cards_with_color = [
                i for i in range(n)
                if color in props['color_identity'].get(i, [])
            ]
            if cards_with_color:
                lb = max(1, int(60 * scale))
                ub = max(lb + 1, int(80 * scale))
                prob += (
                    lpSum(choices[i] for i in cards_with_color) >= lb,
                    f"color_{color}_lb",
                )
                prob += (
                    lpSum(choices[i] for i in cards_with_color) <= ub,
                    f"color_{color}_ub",
                )

        # Colorless cards
        colorless_cards = [
            i for i in range(n)
            if not props['color_identity'].get(i, [])
        ]
        if colorless_cards:
            lb_cl = max(0, int(20 * scale))
            ub_cl = max(lb_cl + 1, int(40 * scale))
            prob += (
                lpSum(choices[i] for i in colorless_cards) >= lb_cl,
                "colorless_lb",
            )
            prob += (
                lpSum(choices[i] for i in colorless_cards) <= ub_cl,
                "colorless_ub",
            )

    # Mana curve constraints (only for set_size >= 20)
    if 'cmc' in props and set_size >= 20:
        low_cmc = [i for i in range(n) if props['cmc'].get(i, 0) <= 2]
        high_cmc = [i for i in range(n) if props['cmc'].get(i, 0) >= 6]
        if low_cmc:
            prob += (
                lpSum(choices[i] for i in low_cmc) >= max(1, int(100 * scale)),
                "low_cmc_lb",
            )
        if high_cmc:
            prob += (
                lpSum(choices[i] for i in high_cmc) <= max(1, int(40 * scale)),
                "high_cmc_ub",
            )

    # Card type constraints (only for set_size >= 20)
    if 'type_line' in props and set_size >= 20:
        creatures = [
            i for i in range(n)
            if 'Creature' in props['type_line'].get(i, '')
        ]
        instants_sorceries = [
            i for i in range(n)
            if 'Instant' in props['type_line'].get(i, '')
            or 'Sorcery' in props['type_line'].get(i, '')
        ]
        lands = [
            i for i in range(n)
            if 'Land' in props['type_line'].get(i, '')
        ]

        if creatures:
            prob += (
                lpSum(choices[i] for i in creatures) >= max(1, int(120 * scale)),
                "creatures_lb",
            )
        if instants_sorceries:
            prob += (
                lpSum(choices[i] for i in instants_sorceries) >= max(1, int(60 * scale)),
                "spells_lb",
            )
        if lands:
            prob += (
                lpSum(choices[i] for i in lands) <= max(1, int(50 * scale)),
                "lands_ub",
            )

    # Theme quota constraints (only for set_size >= 60)
    if 'themes' in props and set_size >= 60:
        from card_metadata import THEME_PATTERNS
        theme_sets = {theme: set() for theme in THEME_PATTERNS}
        themes_dict = props.get('themes', {})
        for idx, card_themes in themes_dict.items():
            for theme in card_themes:
                theme_sets[theme].add(idx)

        theme_quota = max(1, int(15 * scale))
        for theme_name, card_set in theme_sets.items():
            if len(card_set) >= theme_quota * 2:
                prob += (
                    lpSum(choices[i] for i in card_set) >= theme_quota,
                    f"theme_{theme_name}_lb",
                )


# ---------------------------------------------------------------------------
# CP-SAT Solver
# ---------------------------------------------------------------------------

def find_good_subset_cpsat(matrix, set_size, time_limit=120, threads=0):
    """
    Find optimal subset using OR-Tools CP-SAT solver.

    CP-SAT is the strongest open-source solver for binary combinatorial
    problems, typically 5-10x faster than CBC on our problem structure.
    Requires integer coefficients (synergy floats are scaled by 10000).

    Args:
        matrix: Symmetrized sparse synergy matrix (n x n)
        set_size: Number of cards to select
        time_limit: Solver time limit in seconds
        threads: Number of worker threads (0 = auto)

    Returns:
        SolverResult with quality metrics (including exact gap from CP-SAT)
    """
    from ortools.sat.python import cp_model

    t0 = time.time()
    n = matrix.shape[0]
    SCALE = 10000  # float-to-int scaling

    console.print(Markdown("## CP-SAT Solver"))
    console.print(f"  Matrix: {n}x{n}, {matrix.nnz} non-zero entries")
    console.print(f"  Selecting {set_size} cards, time limit {time_limit}s")
    console.print(f"  Threads: {'auto' if threads == 0 else threads}")

    model = cp_model.CpModel()
    x = [model.new_bool_var(f'x_{i}') for i in range(n)]
    model.add(sum(x) == set_size)

    # Build objective from upper triangle of synergy matrix
    matrix_coo = matrix.tocoo()
    obj_terms = []
    num_products = 0

    for k_idx in range(len(matrix_coo.data)):
        i, j = int(matrix_coo.row[k_idx]), int(matrix_coo.col[k_idx])
        if i < j:
            coeff = int(round(float(matrix_coo.data[k_idx]) * SCALE))
            if coeff == 0:
                continue
            y = model.new_bool_var(f'y_{i}_{j}')
            # Linearization: y = x_i AND x_j
            model.add(y <= x[i])
            model.add(y <= x[j])
            model.add(y >= x[i] + x[j] - 1)
            obj_terms.append(coeff * y)
            num_products += 1

    console.print(f"  Product variables: {num_products}")
    model.maximize(sum(obj_terms))

    cpsat_solver = cp_model.CpSolver()
    cpsat_solver.parameters.max_time_in_seconds = time_limit
    if threads > 0:
        cpsat_solver.parameters.num_workers = threads

    status = cpsat_solver.solve(model)
    wall_time = time.time() - t0

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        indices = [i for i in range(n) if cpsat_solver.value(x[i])]
        obj_val = cpsat_solver.objective_value / SCALE
        bound = cpsat_solver.best_objective_bound / SCALE
        if abs(obj_val) > 1e-10:
            gap = abs(bound - obj_val) / abs(obj_val)
        else:
            gap = 0.0
        status_str = 'optimal' if status == cp_model.OPTIMAL else 'feasible'
    else:
        indices = []
        obj_val = 0.0
        bound = None
        gap = None
        status_str = ('infeasible' if status == cp_model.INFEASIBLE
                      else 'unknown')

    result = SolverResult(
        indices=indices,
        score=obj_val,
        method='cpsat',
        wall_time=wall_time,
        status=status_str,
        gap=gap,
        best_bound=bound,
    )
    _print_solver_report(result)
    return result


# ---------------------------------------------------------------------------
# Greedy + Local Search
# ---------------------------------------------------------------------------

def _compute_synergy_score(matrix, selected_indices):
    """Compute total pairwise synergy for a set of card indices (upper triangle)."""
    if len(selected_indices) == 0:
        return 0.0
    idx = np.array(selected_indices)
    submatrix = matrix[np.ix_(idx, idx)]
    if sp.issparse(submatrix):
        submatrix = submatrix.toarray()
    return np.triu(submatrix, k=1).sum()


def find_good_subset_greedy(matrix, set_size, card_properties=None,
                            num_restarts=5, max_no_improve=1000, seed=None):
    """
    Greedy construction + local search for finding good subsets.

    Scales to k=360 from a pool of 5000+ cards. Each step is O(k) for
    swap evaluation, O(k*n) per improvement pass.

    Algorithm:
    1. Random seed (or one card per spectral cluster if available)
    2. Greedy expansion: add card with highest marginal synergy gain
    3. Local search: randomly try swaps, accept if they improve objective

    This is a heuristic — there is no optimality gap or bound.
    Quality knobs: num_restarts and max_no_improve. More of either = better
    solutions but more time. See APPROXIMATION_GUIDE.md.

    Args:
        matrix: Symmetrized sparse synergy matrix
        set_size: Number of cards to select
        card_properties: Optional metadata (not used for constraints yet)
        num_restarts: Number of random restarts
        max_no_improve: Stop local search after this many consecutive non-improving swaps
        seed: Random seed for reproducibility

    Returns:
        SolverResult (status='heuristic', gap=None)
    """
    t0 = time.time()

    if seed is not None:
        np.random.seed(seed)

    n = matrix.shape[0]

    console.print(Markdown("## Greedy + Local Search"))
    console.print(f"  Matrix: {n}x{n}, selecting {set_size} cards")
    console.print(f"  Restarts: {num_restarts}, max_no_improve: {max_no_improve}")

    # Pre-compute dense matrix for faster random access if feasible
    if n <= 5000:
        dense = matrix.toarray() if sp.issparse(matrix) else np.array(matrix)
    else:
        dense = None

    def contribution(card, selected_list):
        """Sum of synergy between card and all cards in selected_list."""
        if len(selected_list) == 0:
            return 0.0
        others = np.array(selected_list)
        if dense is not None:
            return dense[card, others].sum()
        row = matrix[card, others]
        if sp.issparse(row):
            return row.sum()
        return np.sum(row)

    best_score = -np.inf
    best_solution = None
    total_swaps = 0

    for restart in tqdm(range(num_restarts), desc="Greedy restarts"):
        # Phase 1: Greedy construction
        selected = []
        remaining = set(range(n))

        # Random first card
        first = np.random.choice(list(remaining))
        selected.append(first)
        remaining.discard(first)

        # Greedily add the card with highest marginal gain
        while len(selected) < set_size:
            best_gain = -np.inf
            best_card = None

            # Sample candidates if pool is large (avoids O(n) per step)
            if len(remaining) > 1000:
                candidates = np.random.choice(
                    list(remaining), size=1000, replace=False,
                )
            else:
                candidates = list(remaining)

            for card in candidates:
                gain = contribution(card, selected)
                if gain > best_gain:
                    best_gain = gain
                    best_card = card

            selected.append(best_card)
            remaining.discard(best_card)

        # Phase 2: Local search (swap improvement)
        remaining_list = list(remaining)
        current_score = _compute_synergy_score(matrix, selected)
        no_improve_count = 0

        while no_improve_count < max_no_improve:
            # Pick random card to swap out
            idx_in = np.random.randint(len(selected))
            card_in = selected[idx_in]

            # Pick random candidate to swap in
            idx_out = np.random.randint(len(remaining_list))
            card_out = remaining_list[idx_out]

            # Compute swap delta
            others = [c for c in selected if c != card_in]
            loss = contribution(card_in, others)
            gain = contribution(card_out, others)
            delta = gain - loss

            total_swaps += 1
            if delta > 0:
                # Accept swap
                selected[idx_in] = card_out
                remaining_list[idx_out] = card_in
                current_score += delta
                no_improve_count = 0
            else:
                no_improve_count += 1

        if current_score > best_score:
            best_score = current_score
            best_solution = list(selected)

    wall_time = time.time() - t0

    result = SolverResult(
        indices=best_solution,
        score=best_score,
        method='greedy',
        wall_time=wall_time,
        status='heuristic',
        gap=None,
        best_bound=None,
        iterations=total_swaps,
    )
    _print_solver_report(result)
    return result


# ---------------------------------------------------------------------------
# Spectral Analysis
# ---------------------------------------------------------------------------

def spectral_analysis(matrix, n_components=10):
    """
    Compute spectral decomposition of the synergy matrix.

    The leading eigenvectors represent dominant synergy patterns (archetypes).

    Args:
        matrix: Symmetrized sparse synergy matrix
        n_components: Number of eigenvectors to compute

    Returns:
        eigenvalues: Array of top eigenvalues (descending)
        eigenvectors: Matrix of corresponding eigenvectors (n x n_components)
    """
    console.print(Markdown("## Spectral Analysis"))

    S = matrix.astype(np.float64)
    n_components = min(n_components, matrix.shape[0] - 2)
    eigenvalues, eigenvectors = eigsh(S, k=n_components, which='LM')

    # Sort by eigenvalue descending
    order = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    console.print(f"  Top eigenvalues: {np.array2string(eigenvalues, precision=2)}")

    return eigenvalues, eigenvectors


def spectral_cluster(matrix, n_clusters=8, n_components=20):
    """
    Cluster cards by synergy patterns using spectral clustering.

    Identifies natural archetypes in the synergy graph.

    Args:
        matrix: Symmetrized sparse synergy matrix
        n_clusters: Number of clusters (archetypes) to find
        n_components: Number of spectral components to use

    Returns:
        labels: Cluster assignment per card
        cluster_sizes: Dict of cluster_id -> count
    """
    from sklearn.cluster import SpectralClustering

    console.print(Markdown("## Spectral Clustering"))
    console.print(f"  Finding {n_clusters} clusters using {n_components} components")

    S = matrix.astype(np.float64)

    # SpectralClustering needs a non-negative affinity matrix
    if sp.issparse(S):
        min_val = S.min()
    else:
        min_val = np.min(S)
    if min_val < 0:
        S = S - min_val

    if sp.issparse(S):
        S_dense = S.toarray()
    else:
        S_dense = S

    n_components = min(n_components, S_dense.shape[0] - 2)

    clustering = SpectralClustering(
        n_clusters=n_clusters,
        n_components=n_components,
        affinity='precomputed',
        random_state=42,
        assign_labels='kmeans',
    )
    labels = clustering.fit_predict(S_dense)

    cluster_sizes = {}
    for label in range(n_clusters):
        cluster_sizes[label] = int(np.sum(labels == label))

    console.print(f"  Cluster sizes: {cluster_sizes}")

    return labels, cluster_sizes


def spectral_prefilter(matrix, target_n, n_components=20):
    """
    Select diverse candidates using spectral methods.

    Instead of taking the top cards by global synergy (which biases toward
    popular cards), project into spectral space and select cards that span
    the principal components well using farthest-point sampling.

    Args:
        matrix: Symmetrized synergy matrix
        target_n: Number of candidates to select
        n_components: Number of spectral dimensions

    Returns:
        selected_indices: Array of selected card indices
    """
    console.print(Markdown("## Spectral Pre-filtering"))

    n = matrix.shape[0]
    if target_n >= n:
        return np.arange(n)

    n_components = min(n_components, n - 2)

    S = matrix.astype(np.float64)
    eigenvalues, eigenvectors = eigsh(S, k=n_components, which='LM')

    # Project cards into spectral space
    projections = eigenvectors  # n x n_components

    # Farthest-point sampling for diversity
    selected = []

    # Start with the card closest to the centroid
    centroid = projections.mean(axis=0)
    dists_to_centroid = np.linalg.norm(projections - centroid, axis=1)
    first = np.argmin(dists_to_centroid)
    selected.append(first)

    # Track minimum distance from each point to any selected point
    min_dists = np.linalg.norm(projections - projections[first], axis=1)

    for _ in tqdm(range(target_n - 1), desc="Spectral pre-filter"):
        # Select the point farthest from all selected points
        next_idx = np.argmax(min_dists)
        selected.append(next_idx)

        new_dists = np.linalg.norm(projections - projections[next_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    console.print(f"  Selected {len(selected)} diverse candidates")
    return np.array(selected)


# ---------------------------------------------------------------------------
# Log-Synergy Objective (v1)
# ---------------------------------------------------------------------------
#
# Objective: f(C) = sum_{A in C} ln(1 + incoming(A))
#                  - lambda_color * P_color(C)
#                  - lambda_type * P_type(C)
#                  - lambda_curve * P_curve(C)
#
# See notes/ai/OBJECTIVE_V1.md for full spec.

# Default targets from OBJECTIVE_V1.md
COLOR_TARGETS = {
    'W': 0.18, 'U': 0.18, 'B': 0.18, 'R': 0.18, 'G': 0.18,
    'colorless': 0.10,
}
TYPE_TARGETS = {'creature': 0.45, 'noncreature': 0.45, 'land': 0.10}
CURVE_TARGETS = {
    '0-1': 0.15, '2': 0.22, '3': 0.22, '4': 0.18, '5': 0.12, '6+': 0.11,
}


def _classify_type(type_line):
    """Classify a card as 'creature', 'land', or 'noncreature'."""
    if 'Land' in type_line:
        return 'land'
    if 'Creature' in type_line:
        return 'creature'
    return 'noncreature'


def _cmc_bucket(cmc):
    """Map CMC value to a bucket key."""
    c = int(cmc)
    if c <= 1:
        return '0-1'
    if c >= 6:
        return '6+'
    return str(c)


def load_pool_metadata(card_names, card_metadata_path='data/card_metadata.json'):
    """
    Load card metadata indexed by pool position.

    Args:
        card_names: list of sanitized card name slugs
        card_metadata_path: path to card_metadata.json

    Returns:
        dict mapping card_index -> {color_identity, cmc, type_class, name}
    """
    meta_path = Path(card_metadata_path)
    with meta_path.open('r') as f:
        all_meta = json.load(f)

    # Build lookup by sanitized name
    meta_by_name = {}
    for entry in all_meta:
        meta_by_name[entry['sanitized']] = entry

    pool_meta = {}
    matched = 0
    for idx, name in enumerate(card_names):
        entry = meta_by_name.get(name)
        if entry is None:
            continue
        matched += 1
        pool_meta[idx] = {
            'color_identity': entry.get('color_identity', []),
            'cmc': entry.get('cmc', 0.0),
            'type_class': _classify_type(entry.get('type_line', '')),
            'type_line': entry.get('type_line', ''),
            'name': entry.get('name', name),
        }

    console.print(f"  Matched {matched}/{len(card_names)} cards to metadata")
    return pool_meta


def _compute_color_penalty(cube_indices, k, color_arr, colorless_arr, targets):
    """Quadratic color balance penalty."""
    penalty = 0.0
    for color, arr in color_arr.items():
        frac = arr[cube_indices].sum() / k
        penalty += (frac - targets[color]) ** 2
    frac_cl = colorless_arr[cube_indices].sum() / k
    penalty += (frac_cl - targets['colorless']) ** 2
    return penalty


def _compute_type_penalty(cube_indices, k, type_arr, targets):
    """Quadratic type balance penalty."""
    types = type_arr[cube_indices]
    c_count = np.sum(types == 0)  # creature
    l_count = np.sum(types == 2)  # land
    nc_count = k - c_count - l_count  # noncreature
    penalty = ((c_count / k - targets['creature']) ** 2
               + (nc_count / k - targets['noncreature']) ** 2
               + (l_count / k - targets['land']) ** 2)
    return penalty


def _compute_curve_penalty(cube_indices, k, cmc_arr, type_arr, targets):
    """Quadratic mana curve penalty (nonland cards only)."""
    types = type_arr[cube_indices]
    nonland_mask = types != 2  # not land
    nonland_cmcs = cmc_arr[cube_indices][nonland_mask]
    n_nonland = len(nonland_cmcs)
    if n_nonland == 0:
        return 0.0

    penalty = 0.0
    for bucket, target in targets.items():
        if bucket == '0-1':
            count = np.sum(nonland_cmcs <= 1)
        elif bucket == '6+':
            count = np.sum(nonland_cmcs >= 6)
        else:
            count = np.sum(np.floor(nonland_cmcs).astype(int) == int(bucket))
        penalty += (count / n_nonland - target) ** 2
    return penalty


def solve_cube(incoming_matrix, card_names, pool_meta, k,
               lambdas=None, use_outer_log=True,
               num_restarts=5, max_no_improve=2000, seed=42,
               solver='cbc', time_limit=120, gap_rel=0.05, threads=0):
    """
    Find the best cube, auto-selecting the best algorithm.

    - Linear objective + small pool + no penalties → ILP (exact solution)
    - Otherwise → greedy + local search

    Returns: (cube_indices, score, details)
    """
    if lambdas is None:
        lambdas = {'color': 0.0, 'type': 0.0, 'curve': 0.0}

    n = incoming_matrix.shape[0]
    has_penalties = any(v != 0 for v in lambdas.values())

    # Auto-select ILP for linear objective with small pool and no soft penalties
    # (ILP can't handle soft quadratic penalties or log objective)
    # Threshold N<=200: PuLP/CBC model construction is slow for larger pools
    if not use_outer_log and not has_penalties and n <= 200:
        console.print(Markdown("## Strategy: ILP (linear objective, exact)"))
        return _solve_cube_ilp(incoming_matrix, k, solver=solver,
                               time_limit=time_limit, gap_rel=gap_rel,
                               threads=threads)

    objective_name = "log-synergy" if use_outer_log else "linear synergy"
    console.print(Markdown(f"## Strategy: Greedy + Local Search ({objective_name})"))

    return find_best_cube_log_synergy(
        incoming_matrix, card_names, pool_meta, k,
        lambdas=lambdas, use_outer_log=use_outer_log,
        num_restarts=num_restarts, max_no_improve=max_no_improve, seed=seed)


def _solve_cube_ilp(incoming_matrix, k, solver='cbc', time_limit=120,
                    gap_rel=0.05, threads=0):
    """
    Solve using ILP on the symmetrized incoming matrix.

    Returns: (cube_indices, score, details)
    """
    S_sym = symmetrize_matrix(incoming_matrix)

    if solver == 'cpsat':
        ilp_result = find_good_subset_cpsat(S_sym, k, time_limit=time_limit,
                                            threads=threads)
    else:
        ilp_result = find_good_subset_ilp(S_sym, k, solver=solver,
                                          time_limit=time_limit,
                                          gap_rel=gap_rel, threads=threads)
    solution_indices = ilp_result.indices

    # Evaluate with the asymmetric incoming matrix for consistent scoring
    M = incoming_matrix.toarray() if sp.issparse(incoming_matrix) else np.array(incoming_matrix)
    np.fill_diagonal(M, 0)
    idx = np.array(solution_indices)
    sub = M[np.ix_(idx, idx)]
    incoming = sub.sum(axis=0)
    synergy_term = float(np.sum(incoming))

    return list(idx), synergy_term, {
        'synergy_term': synergy_term,
        'color_penalty': 0.0,
        'type_penalty': 0.0,
        'curve_penalty': 0.0,
        'incoming': incoming,
        'solver_result': ilp_result,
    }


def find_best_cube_log_synergy(incoming_matrix, card_names, pool_meta, k,
                               lambdas=None, use_outer_log=True,
                               num_restarts=5, max_no_improve=2000, seed=42):
    """
    Greedy construction + local search optimizing synergy objective.

    When use_outer_log=True (default):
        f(C) = sum_{A in C} ln(1 + incoming(A)) - penalty terms
    When use_outer_log=False:
        f(C) = sum_{A in C} incoming(A) - penalty terms

    Args:
        incoming_matrix: ReLU'd incoming synergy matrix (N x N sparse).
            M[b, a] = max(0, S[b, a]) = how much card b wants card a.
        card_names: list of card name slugs (length N)
        pool_meta: dict from load_pool_metadata()
        k: number of cards to select
        lambdas: dict with 'color', 'type', 'curve' penalty weights
        use_outer_log: if True, apply ln(1+x) to per-card incoming synergy
        num_restarts: random restarts
        max_no_improve: stop local search after this many non-improving swaps
        seed: random seed

    Returns:
        (best_cube_indices, best_score, best_details)
    """
    if lambdas is None:
        lambdas = {'color': 0.0, 'type': 0.0, 'curve': 0.0}

    # The "term" function: log(1+x) for diminishing returns, or identity for linear
    term = (lambda x: np.log(1.0 + x)) if use_outer_log else (lambda x: x)

    n = incoming_matrix.shape[0]
    M = incoming_matrix.toarray() if sp.issparse(incoming_matrix) else np.array(incoming_matrix)
    np.fill_diagonal(M, 0)  # no self-synergy

    # Precompute per-card property arrays for fast vectorized lookups
    # color_arr: dict of color -> bool array
    color_arr = {}
    for color in ['W', 'U', 'B', 'R', 'G']:
        arr = np.zeros(n, dtype=bool)
        for idx, meta in pool_meta.items():
            if color in meta['color_identity']:
                arr[idx] = True
        color_arr[color] = arr

    colorless_arr = np.zeros(n, dtype=bool)
    for idx, meta in pool_meta.items():
        if not meta['color_identity']:
            colorless_arr[idx] = True

    # type_arr: 0=creature, 1=noncreature, 2=land
    type_arr = np.ones(n, dtype=int)  # default noncreature
    for idx, meta in pool_meta.items():
        if meta['type_class'] == 'creature':
            type_arr[idx] = 0
        elif meta['type_class'] == 'land':
            type_arr[idx] = 2

    cmc_arr = np.zeros(n)
    for idx, meta in pool_meta.items():
        cmc_arr[idx] = meta['cmc']

    has_penalties = (lambdas['color'] != 0 or lambdas['type'] != 0
                     or lambdas['curve'] != 0)

    obj_label = "Log-Synergy" if use_outer_log else "Linear Synergy"
    console.print(Markdown(f"## {obj_label} Greedy + Local Search"))
    console.print(f"  Pool: {n} cards, selecting {k}")
    console.print(f"  Objective: {'ln(1 + incoming)' if use_outer_log else 'sum(incoming)'}")
    console.print(f"  Restarts: {num_restarts}, max_no_improve: {max_no_improve}")
    console.print(f"  Lambdas: color={lambdas['color']}, type={lambdas['type']}, "
                  f"curve={lambdas['curve']}")

    t_start = time.time()

    # Column sums = total incoming synergy per card (used for seeding)
    col_sums = M.sum(axis=0)

    best_score = -np.inf
    best_cube = None
    best_details = None
    best_restart_idx = 0
    total_swaps_all = 0

    for restart in tqdm(range(num_restarts), desc="Restarts"):
        rng = np.random.RandomState(seed + restart)

        # ------------------------------------------------------------------
        # Phase 1: Greedy construction
        # ------------------------------------------------------------------
        in_cube = np.zeros(n, dtype=bool)

        if restart == 0:
            # First restart: seed with top incoming-synergy card
            first = int(np.argmax(col_sums))
        else:
            # Subsequent restarts: weighted random by column sum
            weights = np.maximum(col_sums, 0)
            total_w = weights.sum()
            if total_w > 0:
                weights = weights / total_w
            else:
                weights = np.ones(n) / n
            first = int(rng.choice(n, p=weights))

        cube = [first]
        in_cube[first] = True

        # Track incoming synergy for each card in the cube
        # incoming_vals[pos] = sum of M[b, cube[pos]] for b in cube, b != cube[pos]
        incoming_vals = [0.0]

        while len(cube) < k:
            cube_arr = np.array(cube)
            candidates = np.where(~in_cube)[0]

            # Incoming synergy each candidate would receive from current cube
            # candidate_incoming[i] = sum of M[b, candidates[i]] for b in cube
            candidate_incoming = M[cube_arr][:, candidates].sum(axis=0)

            # Delta for existing cards when adding each candidate
            # M[candidates][:, cube_arr] gives contributions FROM each candidate TO cube cards
            inc_arr = np.array(incoming_vals)
            contrib = M[candidates][:, cube_arr]  # (n_candidates, len(cube))

            old_terms = term(inc_arr)  # (len(cube),)
            new_terms = term(inc_arr[np.newaxis, :] + contrib)  # (n_cand, len(cube))
            delta_existing = (new_terms - old_terms[np.newaxis, :]).sum(axis=1)

            # Total gain per candidate
            gains = term(candidate_incoming) + delta_existing

            best_idx = int(np.argmax(gains))
            best_card = int(candidates[best_idx])

            # Update incoming for existing cube cards
            for pos in range(len(cube)):
                incoming_vals[pos] += M[best_card, cube[pos]]

            # Incoming for the new card
            incoming_vals.append(float(candidate_incoming[best_idx]))

            cube.append(best_card)
            in_cube[best_card] = True

        # ------------------------------------------------------------------
        # Phase 2: Local search (random swaps)
        # ------------------------------------------------------------------
        cube_arr = np.array(cube, dtype=int)

        # Recompute accurate incoming from scratch (avoids float drift)
        sub = M[np.ix_(cube_arr, cube_arr)]
        incoming_cube = sub.sum(axis=0).copy()  # shape (k,)

        synergy_score = float(np.sum(term(incoming_cube)))
        if has_penalties:
            pen_color = _compute_color_penalty(
                cube_arr, k, color_arr, colorless_arr, COLOR_TARGETS)
            pen_type = _compute_type_penalty(cube_arr, k, type_arr, TYPE_TARGETS)
            pen_curve = _compute_curve_penalty(
                cube_arr, k, cmc_arr, type_arr, CURVE_TARGETS)
            current_score = (synergy_score
                             - lambdas['color'] * pen_color
                             - lambdas['type'] * pen_type
                             - lambdas['curve'] * pen_curve)
        else:
            current_score = synergy_score

        pool = np.where(~in_cube)[0].tolist()

        no_improve = 0
        swaps_accepted = 0

        while no_improve < max_no_improve:
            out_pos = rng.randint(k)
            card_out = cube_arr[out_pos]
            in_idx = rng.randint(len(pool))
            card_in = pool[in_idx]

            # --- Synergy delta ---
            # Mask for "other" cards (not the one being swapped out)
            other_positions = np.arange(k) != out_pos
            other_indices = cube_arr[other_positions]

            # Remove card_out's synergy term
            delta = -float(term(incoming_cube[out_pos]))

            # For other cards: incoming changes by -M[card_out, a] + M[card_in, a]
            old_inc_others = incoming_cube[other_positions]
            delta_inc = M[card_in, other_indices] - M[card_out, other_indices]
            new_inc_others = old_inc_others + delta_inc
            delta += float(np.sum(term(new_inc_others) - term(old_inc_others)))

            # Add card_in's synergy term
            # incoming(card_in) = sum of M[b, card_in] for b in new cube (excl card_in)
            card_in_incoming = float(M[other_indices, card_in].sum())
            delta += float(term(card_in_incoming))

            # --- Penalty delta (only computed when lambdas are nonzero) ---
            if has_penalties:
                # Build proposed cube for penalty re-evaluation
                proposed = cube_arr.copy()
                proposed[out_pos] = card_in
                new_pen_color = _compute_color_penalty(
                    proposed, k, color_arr, colorless_arr, COLOR_TARGETS)
                new_pen_type = _compute_type_penalty(
                    proposed, k, type_arr, TYPE_TARGETS)
                new_pen_curve = _compute_curve_penalty(
                    proposed, k, cmc_arr, type_arr, CURVE_TARGETS)
                delta -= lambdas['color'] * (new_pen_color - pen_color)
                delta -= lambdas['type'] * (new_pen_type - pen_type)
                delta -= lambdas['curve'] * (new_pen_curve - pen_curve)

            if delta > 1e-12:
                # Accept swap
                pool[in_idx] = int(card_out)
                in_cube[card_out] = False
                in_cube[card_in] = True
                cube_arr[out_pos] = card_in

                # Update incoming vector
                incoming_cube[other_positions] = new_inc_others
                incoming_cube[out_pos] = card_in_incoming

                current_score += delta
                swaps_accepted += 1
                no_improve = 0

                if has_penalties:
                    pen_color = new_pen_color
                    pen_type = new_pen_type
                    pen_curve = new_pen_curve
            else:
                no_improve += 1

        # Compute final score from scratch for accuracy
        final_score, details = _evaluate_log_synergy_full(
            M, cube_arr, k, color_arr, colorless_arr, type_arr, cmc_arr,
            lambdas, use_outer_log=use_outer_log)

        total_swaps_all += swaps_accepted

        tqdm.write(f"  Restart {restart}: score={final_score:.4f}, "
                   f"synergy={details['synergy_term']:.4f}, "
                   f"swaps={swaps_accepted}")

        if final_score > best_score:
            best_score = final_score
            best_cube = cube_arr.copy()
            best_details = details
            best_restart_idx = restart

    wall_time = time.time() - t_start

    console.print(f"\n  Best score: {best_score:.4f} (restart {best_restart_idx})")

    # Attach solver metadata for reporting
    best_details['solver_result'] = SolverResult(
        indices=list(best_cube),
        score=best_score,
        method='greedy',
        wall_time=wall_time,
        status='heuristic',
        gap=None,
        best_bound=None,
        iterations=total_swaps_all,
    )

    return list(best_cube), best_score, best_details


def _evaluate_log_synergy_full(M, cube_indices, k, color_arr, colorless_arr,
                               type_arr, cmc_arr, lambdas, use_outer_log=True):
    """
    Full evaluation of synergy objective for a cube.

    Returns (score, details_dict).
    """
    term = (lambda x: np.log(1.0 + x)) if use_outer_log else (lambda x: x)
    idx = np.array(cube_indices)
    sub = M[np.ix_(idx, idx)]
    incoming = sub.sum(axis=0)
    synergy_term = float(np.sum(term(incoming)))

    pen_color = _compute_color_penalty(idx, k, color_arr, colorless_arr, COLOR_TARGETS)
    pen_type = _compute_type_penalty(idx, k, type_arr, TYPE_TARGETS)
    pen_curve = _compute_curve_penalty(idx, k, cmc_arr, type_arr, CURVE_TARGETS)

    score = (synergy_term
             - lambdas['color'] * pen_color
             - lambdas['type'] * pen_type
             - lambdas['curve'] * pen_curve)

    return score, {
        'synergy_term': synergy_term,
        'color_penalty': pen_color,
        'type_penalty': pen_type,
        'curve_penalty': pen_curve,
        'incoming': incoming,
    }


def display_cube_results(cube_indices, card_names, pool_meta, score, details,
                         M_dense, lambdas, use_outer_log=True):
    """Pretty-print the selected cube with stats."""
    term = (lambda x: np.log(1.0 + x)) if use_outer_log else (lambda x: x)
    k = len(cube_indices)
    idx = np.array(cube_indices)

    console.print(Markdown("---"))
    obj_label = "Log-Synergy" if use_outer_log else "Linear Synergy"
    console.print(Markdown(f"## Results ({obj_label})"))
    console.print()

    # Score breakdown
    console.print(f"  Total score: {score:.4f}")
    console.print(f"    Synergy term: {details['synergy_term']:.4f}")
    console.print(f"    Color penalty: {details['color_penalty']:.4f} "
                  f"(lambda={lambdas['color']})")
    console.print(f"    Type penalty: {details['type_penalty']:.4f} "
                  f"(lambda={lambdas['type']})")
    console.print(f"    Curve penalty: {details['curve_penalty']:.4f} "
                  f"(lambda={lambdas['curve']})")
    console.print()

    # Card table
    incoming = details['incoming']
    term_col = "ln(1+inc)" if use_outer_log else "raw inc"
    table = Table(title=f"Selected Cube (K={k})")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Card Name", style="bold")
    table.add_column("Colors")
    table.add_column("Type")
    table.add_column("CMC", justify="right")
    table.add_column("Incoming Syn", justify="right")
    table.add_column(term_col, justify="right")

    # Sort by incoming synergy descending
    order = np.argsort(-incoming)
    for rank, pos in enumerate(order):
        card_idx = cube_indices[pos]
        meta = pool_meta.get(card_idx, {})
        name = meta.get('name', card_names[card_idx])
        colors = ''.join(meta.get('color_identity', [])) or 'C'
        type_line = meta.get('type_line', '')
        # Shorten type_line
        type_short = type_line.split('—')[0].strip()
        if len(type_short) > 25:
            type_short = type_short[:22] + '...'
        cmc = meta.get('cmc', 0)
        inc = incoming[pos]
        term_val = float(term(inc))

        table.add_row(
            str(rank + 1),
            name,
            colors,
            type_short,
            f"{cmc:.0f}",
            f"{inc:.3f}",
            f"{term_val:.3f}",
        )

    console.print(table)
    console.print()

    # Color distribution
    color_counts = {'W': 0, 'U': 0, 'B': 0, 'R': 0, 'G': 0, 'C': 0}
    for card_idx in cube_indices:
        meta = pool_meta.get(card_idx, {})
        ci = meta.get('color_identity', [])
        if not ci:
            color_counts['C'] += 1
        for c in ci:
            if c in color_counts:
                color_counts[c] += 1

    console.print(f"  Color distribution: {color_counts}")

    # Type distribution
    type_counts = {'creature': 0, 'noncreature': 0, 'land': 0}
    for card_idx in cube_indices:
        meta = pool_meta.get(card_idx, {})
        tc = meta.get('type_class', 'noncreature')
        type_counts[tc] += 1

    console.print(f"  Type distribution: {type_counts}")

    # Mana curve
    curve = {'0-1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6+': 0}
    for card_idx in cube_indices:
        meta = pool_meta.get(card_idx, {})
        if meta.get('type_class') == 'land':
            continue
        bucket = _cmc_bucket(meta.get('cmc', 0))
        curve[bucket] += 1

    console.print(f"  Mana curve (nonland): {curve}")


# ---------------------------------------------------------------------------
# Main pipeline (old linear objective)
# ---------------------------------------------------------------------------

def optimize(synergy_matrix, card_names, set_size=10, method='ilp',
             prefilter='topn', prefilter_multiplier=50, time_limit=120,
             card_properties=None, num_restarts=5, solver='cbc',
             gap_rel=0.05, threads=0):
    """
    Main optimization pipeline.

    1. Symmetrize the raw synergy matrix
    2. Pre-filter candidates (top-N or spectral)
    3. Solve (ILP, CP-SAT, or greedy+local search)
    4. Map results back to original card names

    Args:
        synergy_matrix: Raw (possibly asymmetric) sparse synergy matrix
        card_names: List of card names indexed by matrix position
        set_size: Number of cards to select
        method: 'ilp' or 'greedy'
        prefilter: 'topn' or 'spectral'
        prefilter_multiplier: For topn, keep set_size * multiplier candidates
        time_limit: Solver time limit in seconds
        card_properties: Optional per-card metadata dict for constraints
        num_restarts: For greedy method, number of random restarts
        solver: ILP backend ('cbc', 'highs', 'cpsat')
        gap_rel: Relative optimality gap for ILP (0.05 = 5%)
        threads: Solver threads (0 = auto)

    Returns:
        selected_names: List of selected card names
        selected_original_indices: List of original matrix indices
        score: Objective value (upper-triangle synergy sum)
    """
    console.print(Markdown("# Opticube Optimizer"))
    console.print(f"  Universe: {synergy_matrix.shape[0]} cards")
    console.print(f"  Target: {set_size} cards")
    console.print(f"  Method: {method}, prefilter: {prefilter}")

    # Step 1: Symmetrize
    console.print(Markdown("### Symmetrizing matrix..."))
    S_sym = symmetrize_matrix(synergy_matrix)
    console.print(f"  Original nnz: {synergy_matrix.nnz}, Symmetrized nnz: {S_sym.nnz}")

    # Step 2: Pre-filter to reduce problem size
    if prefilter == 'spectral' and S_sym.shape[0] > set_size * 5:
        target_n = min(set_size * prefilter_multiplier, S_sym.shape[0])
        selected_candidates = spectral_prefilter(S_sym, target_n)
        reduced_matrix = S_sym[np.ix_(selected_candidates, selected_candidates)]
        index_map = selected_candidates
    elif prefilter == 'topn' and S_sym.shape[0] > set_size * prefilter_multiplier:
        top_n = set_size * prefilter_multiplier
        reduced_matrix, index_map = reduce_matrix(S_sym, top_n)
    else:
        reduced_matrix = S_sym
        index_map = np.arange(S_sym.shape[0])

    console.print(f"  Reduced to {reduced_matrix.shape[0]} candidates")

    # Step 3: Remap card properties to reduced indices
    reduced_properties = None
    if card_properties is not None:
        reduced_properties = {}
        for key, prop_dict in card_properties.items():
            reduced_properties[key] = {
                new_idx: prop_dict[int(index_map[new_idx])]
                for new_idx in range(len(index_map))
                if int(index_map[new_idx]) in prop_dict
            }

    # Step 4: Solve
    if method == 'ilp':
        if solver == 'cpsat':
            result = find_good_subset_cpsat(
                reduced_matrix, set_size,
                time_limit=time_limit,
                threads=threads,
            )
        else:
            result = find_good_subset_ilp(
                reduced_matrix, set_size,
                card_properties=reduced_properties,
                time_limit=time_limit,
                gap_rel=gap_rel,
                solver=solver,
                threads=threads,
            )
        solution_indices = result.indices
        score = _compute_synergy_score(reduced_matrix, solution_indices)
    elif method == 'greedy':
        result = find_good_subset_greedy(
            reduced_matrix, set_size,
            card_properties=reduced_properties,
            num_restarts=num_restarts,
        )
        solution_indices = result.indices
        score = result.score
    else:
        raise ValueError(f"Unknown method: {method}")

    # Step 5: Map back to original indices (fixes the index mapping bug)
    original_indices = [int(index_map[i]) for i in solution_indices]
    selected_names = [card_names[i] for i in original_indices]

    # Print results
    console.print(Markdown("### Results"))
    console.print(f"  Synergy score: {score:.2f}")
    for name in sorted(selected_names):
        console.print(f"  - {name}")

    return selected_names, original_indices, score


def main():
    parser = argparse.ArgumentParser(
        description="Opticube Optimizer - Find optimal MTG card subsets by synergy",
    )
    parser.add_argument(
        '--set-size', '-k', type=int, default=10,
        help='Number of cards to select (default: 10)',
    )
    parser.add_argument(
        '--method', choices=['ilp', 'greedy', 'log-synergy'], default='log-synergy',
        help='Solving method (default: log-synergy)',
    )
    parser.add_argument(
        '--prefilter', choices=['topn', 'spectral'], default='topn',
        help='Pre-filtering strategy for ilp/greedy (default: topn)',
    )
    parser.add_argument(
        '--prefilter-multiplier', type=int, default=50,
        help='For topn prefilter: keep set_size * multiplier candidates (default: 50)',
    )
    parser.add_argument(
        '--time-limit', type=int, default=120,
        help='ILP solver time limit in seconds (default: 120)',
    )
    parser.add_argument(
        '--effort', choices=['quick', 'normal', 'thorough', 'max'], default='normal',
        help='Effort preset (quick/normal/thorough/max). '
             'Controls restarts and patience. Override with --num-restarts/--max-no-improve.',
    )
    parser.add_argument(
        '--num-restarts', type=int, default=None,
        help='Number of random restarts (overrides --effort)',
    )
    parser.add_argument(
        '--max-no-improve', type=int, default=None,
        help='Local search: stop after N non-improving swaps (overrides --effort)',
    )
    parser.add_argument(
        '--metric', choices=['synergy', 'lift', 'ppmi'], default='ppmi',
        help='Which EDHREC metric to use (default: ppmi)',
    )
    parser.add_argument(
        '--objective', choices=['log', 'linear'], default='log',
        help='Objective: log = ln(1+incoming), linear = raw sum (default: log)',
    )
    parser.add_argument(
        '--synergy-matrix', type=str, default='synergy_matrix.pkl',
        help='Path to synergy matrix pickle file (for ilp/greedy)',
    )
    parser.add_argument(
        '--incoming-matrix', type=str, default=None,
        help='Override: path to incoming matrix pickle (auto-selected by --metric)',
    )
    parser.add_argument(
        '--card-names', type=str, default='card_names.pkl',
        help='Path to card names pickle file',
    )
    parser.add_argument(
        '--card-metadata', type=str, default='data/card_metadata.json',
        help='Path to card_metadata.json (default: data/card_metadata.json)',
    )
    parser.add_argument(
        '--lambda-color', type=float, default=0.0,
        help='Color balance penalty weight (default: 0.0)',
    )
    parser.add_argument(
        '--lambda-type', type=float, default=0.0,
        help='Type balance penalty weight (default: 0.0)',
    )
    parser.add_argument(
        '--lambda-curve', type=float, default=0.0,
        help='Mana curve penalty weight (default: 0.0)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)',
    )
    parser.add_argument(
        '--scryfall-data', type=str, default=None,
        help='Path to Scryfall oracle cards JSON (for ilp/greedy constraints)',
    )
    parser.add_argument(
        '--spectral-clusters', type=int, default=0,
        help='Run spectral clustering with N clusters (0 = skip)',
    )

    # --- Solver effort / quality knobs ---
    parser.add_argument(
        '--solver', choices=['cbc', 'highs', 'cpsat'], default='cbc',
        help='ILP solver backend (default: cbc). '
             'cpsat is fastest for binary problems. '
             'highs is SOTA for general MIP. '
             'Only used when method is ilp or log-synergy falls back to ILP.',
    )
    parser.add_argument(
        '--gap-rel', type=float, default=0.05,
        help='ILP optimality gap tolerance (default: 0.05 = 5%%). '
             'Set to 0.0 to prove exact optimum. '
             'Only affects ILP solvers (cbc, highs). '
             'CP-SAT always proves optimality when it finishes.',
    )
    parser.add_argument(
        '--threads', type=int, default=0,
        help='Solver threads (default: 0 = auto). '
             'Affects CP-SAT (num_workers) and HiGHS. '
             'CBC is single-threaded.',
    )

    args = parser.parse_args()

    # --- Log-synergy method ---
    if args.method == 'log-synergy':
        console.print(Markdown("# Opticube — Log-Synergy Optimizer"))

        # Determine which incoming matrix to load
        if args.incoming_matrix is not None:
            matrix_path = args.incoming_matrix
        elif args.metric == 'ppmi':
            matrix_path = 'incoming_ppmi_matrix.pkl'
        elif args.metric == 'lift':
            matrix_path = 'incoming_lift_matrix.pkl'
        else:
            matrix_path = 'incoming_synergy_matrix.pkl'

        console.print(Markdown("### Loading data"))
        console.print(f"  Metric: **{args.metric}** (matrix: {matrix_path})")
        with open(matrix_path, 'rb') as f:
            incoming_matrix = pickle.load(f)
        console.print(f"  Incoming matrix: {incoming_matrix.shape}, "
                      f"{incoming_matrix.nnz} non-zero")

        with open(args.card_names, 'rb') as f:
            card_names = pickle.load(f)
        console.print(f"  Card pool: {len(card_names)} cards")

        # Load metadata
        pool_meta = load_pool_metadata(card_names, args.card_metadata)

        lambdas = {
            'color': args.lambda_color,
            'type': args.lambda_type,
            'curve': args.lambda_curve,
        }
        use_outer_log = (args.objective == 'log')
        console.print(f"  Objective: **{'log' if use_outer_log else 'linear'}**")

        # Resolve effort preset (explicit --num-restarts/--max-no-improve override)
        num_restarts, max_no_improve = resolve_effort(
            args.effort, args.num_restarts, args.max_no_improve)
        console.print(f"  Effort: **{args.effort}** "
                      f"({num_restarts} restarts, patience {max_no_improve})")

        # Run optimizer (auto-selects ILP for linear + small pool, else greedy)
        cube_indices, score, details = solve_cube(
            incoming_matrix=incoming_matrix,
            card_names=card_names,
            pool_meta=pool_meta,
            k=args.set_size,
            lambdas=lambdas,
            use_outer_log=use_outer_log,
            num_restarts=num_restarts,
            max_no_improve=max_no_improve,
            seed=args.seed,
            solver=args.solver,
            time_limit=args.time_limit,
            gap_rel=args.gap_rel,
            threads=args.threads,
        )

        # Solver report
        solver_result = details.get('solver_result')
        if solver_result:
            _print_solver_report(solver_result, effort_name=args.effort,
                                 metric=args.metric, objective=args.objective)

        # Display results
        M_dense = incoming_matrix.toarray() if sp.issparse(incoming_matrix) else incoming_matrix
        display_cube_results(
            cube_indices, card_names, pool_meta, score, details, M_dense,
            lambdas, use_outer_log=use_outer_log)

        return

    # --- Old methods (ilp, greedy) ---
    console.print(Markdown("# Loading Data"))

    with open(args.synergy_matrix, 'rb') as f:
        synergy_matrix = pickle.load(f)
    console.print(f"  Synergy matrix: {synergy_matrix.shape}, {synergy_matrix.nnz} non-zero")

    with open(args.card_names, 'rb') as f:
        card_names = pickle.load(f)
    console.print(f"  Card names: {len(card_names)}")

    # Load Scryfall data for constraints (optional)
    card_properties = None
    if args.scryfall_data:
        from card_metadata import load_scryfall_data, build_card_properties
        scryfall_data = load_scryfall_data(args.scryfall_data)
        card_properties = build_card_properties(card_names, scryfall_data)
        console.print("  Loaded Scryfall metadata for constraints")

    # Optional: spectral clustering analysis
    if args.spectral_clusters > 0:
        S_sym = symmetrize_matrix(synergy_matrix)
        labels, cluster_sizes = spectral_cluster(
            S_sym, n_clusters=args.spectral_clusters,
        )
        console.print(Markdown("### Archetype Cluster Samples"))
        for cluster_id in sorted(cluster_sizes.keys()):
            members = np.where(labels == cluster_id)[0]
            row_sums = np.array(S_sym[members].sum(axis=1)).flatten()
            top_local = np.argsort(-row_sums)[:5]
            top_names = [card_names[members[i]] for i in top_local]
            console.print(
                f"  Cluster {cluster_id} ({cluster_sizes[cluster_id]} cards): "
                f"{', '.join(top_names)}"
            )

    # Run optimizer
    optimize(
        synergy_matrix=synergy_matrix,
        card_names=card_names,
        set_size=args.set_size,
        method=args.method,
        prefilter=args.prefilter,
        prefilter_multiplier=args.prefilter_multiplier,
        time_limit=args.time_limit,
        card_properties=card_properties,
        num_restarts=args.num_restarts,
        solver=args.solver,
        gap_rel=args.gap_rel,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
