# Opticube - Mathematical Formulation

## The Core Question

Given a large universe of MTG cards with known pairwise synergy scores, choose exactly `k` cards such that the total synergy among all selected pairs is maximized.

---

## The Synergy Matrix

Let there be `n` cards in the universe. We construct a matrix `S` of size `n x n` where:

```
S[i, j] = synergy percentage of card j relative to card i
```

This value comes from EDHREC and represents how much *more* (or less) often card `j` appears in decks containing card `i` than would be expected by base rate. For example, `S[i, j] = +14` means card `j` appears in 14% more decks containing card `i` than a baseline model would predict.

**Key property: `S` is not symmetric.** The synergy of card `j` relative to card `i` may differ from the synergy of card `i` relative to card `j`, because:
- The data comes from different EDHREC pages (card `i`'s page vs card `j`'s page).
- The reference populations differ (decks containing card `i` vs decks containing card `j`).
- One direction may have data while the other doesn't (the crawl may not have downloaded both pages).

`S` is also **sparse** -- most card pairs have no recorded synergy relationship.

---

## The Objective Function

### What We Want to Maximize

We want to choose a subset `C` of exactly `k` cards from the universe of `n` cards that maximizes:

\[
f(C) = \sum_{i \in C} \sum_{j \in C} S[i, j]
\]

In words: the sum of all pairwise synergy values among the selected cards (including both directions `S[i,j]` and `S[j,i]` when both exist).

This is a **combinatorial optimization** problem. The search space is \(\binom{n}{k}\), which is astronomically large for realistic values (e.g., \(\binom{53000}{10}\)).

---

## Decision Variables

Introduce binary decision variables:

\[
x_i \in \{0, 1\} \quad \text{for each card } i \in \{1, \ldots, n\}
\]

where \(x_i = 1\) means card \(i\) is selected.

The objective becomes:

\[
\text{maximize} \quad \sum_{i=1}^{n} \sum_{j=1}^{n} S[i, j] \cdot x_i \cdot x_j
\]

subject to:

\[
\sum_{i=1}^{n} x_i = k
\]

This is a **Binary Quadratic Program (BQP)** -- the objective is quadratic in the decision variables, and the variables are binary.

---

## Linearization (the ILP Reformulation)

BQPs are hard to solve directly. The project linearizes the problem by introducing auxiliary variables for each product term.

For each pair \((i, j)\) where \(i \leq j\) and \(S[i,j] \neq 0\), introduce:

\[
y_{ij} \in \{0, 1\}
\]

which represents the product \(x_i \cdot x_j\). The constraints that enforce this are:

**For off-diagonal pairs** (\(i \neq j\)):

\[
y_{ij} \leq x_i
\]
\[
y_{ij} \leq x_j
\]
\[
y_{ij} \geq x_i + x_j - 1
\]

The first two constraints ensure \(y_{ij} = 0\) if either card is not selected. The third ensures \(y_{ij} = 1\) when both cards are selected. Together, they enforce \(y_{ij} = x_i \cdot x_j\).

**For diagonal entries** (\(i = j\)):

\[
y_{ii} = x_i
\]

(The diagonal of the synergy matrix represents a card's "synergy with itself," which is just whether the card is selected.)

### The Linearized Objective

\[
\text{maximize} \quad \sum_{(i,j) : i \leq j,\; S[i,j] \neq 0} S[i,j] \cdot y_{ij}
\]

**Note:** The implementation only iterates over pairs where `i <= j`, so it avoids double-counting. However, since `S` is asymmetric, this means only one direction of each pair's synergy is captured. A more complete formulation would symmetrize the matrix first (e.g., `S' = (S + S^T) / 2`) or sum both directions explicitly.

### The Full ILP

\[
\begin{aligned}
\text{maximize} \quad & \sum_{(i,j) \in E} S[i,j] \cdot y_{ij} \\
\text{subject to} \quad & \sum_{i=1}^{n} x_i = k \\
& y_{ij} \leq x_i & \forall (i,j) \in E,\; i \neq j \\
& y_{ij} \leq x_j & \forall (i,j) \in E,\; i \neq j \\
& y_{ij} \geq x_i + x_j - 1 & \forall (i,j) \in E,\; i \neq j \\
& y_{ii} = x_i & \forall (i,i) \in E \\
& x_i \in \{0, 1\} & \forall i \\
& y_{ij} \in \{0, 1\} & \forall (i,j) \in E
\end{aligned}
\]

where \(E = \{(i,j) : i \leq j,\; S[i,j] \neq 0\}\) is the set of non-zero edges.

---

## Problem Reduction Heuristic

The full matrix may be ~53,000 x 53,000. Solving the ILP over all of these is intractable (the number of auxiliary \(y_{ij}\) variables alone could be enormous).

The project applies a **greedy pre-filter**:

1. Compute a score for each card: `score(i) = row_sum(i) + col_sum(i)`, i.e., the total synergy the card has with all other cards in both directions.
2. Keep only the top `m = k * 10` cards by this score.
3. Solve the ILP on the reduced `m x m` submatrix.

This is a heuristic -- it assumes that globally high-synergy cards are likely to be in the optimal subset. It could miss cards that have low total synergy but very high synergy with each other (a tight cluster in a sparse region of the graph).

---

## Solver Configuration

The ILP is solved using PuLP with the CBC (Coin-or Branch and Cut) solver:

- **Time limit:** 10 seconds
- **Relative gap:** 5% (the solver can stop early if it finds a solution within 5% of the proven upper bound)
- **Max solutions:** 1 (stop after finding the first feasible solution)

These are aggressive settings that trade optimality for speed. The solution may not be globally optimal.

---

## Interpretation

The objective function rewards selecting cards that are **mutually synergistic** -- cards that frequently appear together in real Commander decks beyond what chance would predict. A high-scoring subset represents a group of cards where every card "wants to be" in the same deck as every other card.

This is essentially finding a **dense subgraph** in a weighted synergy graph, where:
- Nodes = cards
- Edge weight = synergy percentage
- Goal = maximum-weight k-clique (or more precisely, the densest induced subgraph on exactly k vertices)

---

## Alternative Approaches Attempted

### Random Sampling
Sample random subsets of size `k`, compute \(f(C)\) for each, keep the best. This is a Monte Carlo approach -- simple but with no convergence guarantees. For `num_tries = 1000` and `k = 10`, you're sampling 1000 points from a space of \(\binom{n}{k}\), which is vanishingly small coverage.

### Quadratic Programming (cvxpy)
Attempted to solve the BQP directly using `cvxpy` with `cp.quad_form(x, S)`. This failed because the available solvers (SCS, etc.) don't support mixed-integer quadratic programs. A commercial solver like Gurobi or CPLEX would be needed.

---

## Potential Extensions

- **Symmetrize the matrix** before solving: \(S' = (S + S^T) / 2\), so both directions of synergy are captured.
- **Weight by confidence**: Multiply synergy by `num_decks` or `percentage` to down-weight synergy scores based on small sample sizes.
- **Add cube design constraints**: color identity balance, mana curve distribution, card type ratios.
- **Larger subset sizes**: The current `k = 10` is far from a real cube (typically 360-720 cards). Scaling would require better heuristics or decomposition methods.
- **Iterative refinement**: Solve for a small core, then greedily expand.
