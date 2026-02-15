# Objective Function v1

The first implementable objective. Designed to be simple, theoretically motivated, and buildable with the data we have now (502 EDHREC JSON files, Scryfall metadata for ~53k cards).

---

## The Objective

\[
f(C) = \underbrace{\sum_{A \in C} \ln\!\left(1 + \sum_{\substack{B \in C \\ B \neq A}} \text{ReLU}\!\left(S[B, A]\right)\right)}_{\text{incoming log-synergy}} \;-\; \underbrace{\lambda_{\text{color}} \cdot P_{\text{color}}(C)}_{\text{color balance penalty}} \;-\; \underbrace{\lambda_{\text{type}} \cdot P_{\text{type}}(C)}_{\text{type balance penalty}} \;-\; \underbrace{\lambda_{\text{curve}} \cdot P_{\text{curve}}(C)}_{\text{mana curve penalty}}
\]

Maximize this. Natural log throughout.

---

## Term 1: Incoming Log-Synergy

For each card A in the cube, compute how much the *other* cards in the cube value A:

```
incoming(A) = sum over B in C, B != A: max(0, S[B, A])
```

Then take ln(1 + incoming(A)) and sum across all cards.

**S[B, A]** is the synergy score of card A as reported on card B's EDHREC page. This is the "incoming" direction — it measures how much B wants A around. We use the raw asymmetric matrix; no symmetrization.

**ReLU** (max with 0) discards negative synergy. Anti-synergy is normal and fine — a board wipe and a token maker are anti-synergistic but both belong in good cubes. We don't penalize it; we just don't reward it.

**ln(1 + x)** creates diminishing returns. A card that's highly valued by 50 cards doesn't score 5x more than one valued by 10 cards. This prevents monoculture — piling everything into one dense archetype hits diminishing returns, so the optimizer naturally spreads synergy across multiple clusters.

**Key property:** Two cards each with incoming synergy 50 contribute 2·ln(51) ≈ 7.87. One card with incoming 100 and one with incoming 0 contribute ln(101) + ln(1) ≈ 4.62. The balanced case wins decisively. The log makes the optimizer prefer "every card is wanted" over "a few cards are extremely wanted."

### Data source

`S[B, A]` comes from the EDHREC JSON files. In each card B's JSON (`cards_json/{B}.json`), the `cardlists[*].cardviews[*]` entries contain a `synergy` field for each related card A. This populates the matrix entry S[B, A].

We already build this matrix in `load_cards_files.py`. The synergy matrix is stored as a sparse `scipy.csr_matrix`.

---

## Term 2: Color Balance Penalty

\[
P_{\text{color}}(C) = \sum_{c \in \{W, U, B, R, G\}} \left(\frac{n_c(C)}{|C|} - t_c\right)^2
\]

where:
- `n_c(C)` = number of cards in cube C whose color identity includes color c
- `|C|` = total cards in the cube (e.g. 360)
- `t_c` = target fraction for color c

**Note on multicolor:** A card with color identity {W, U} counts toward both W and U. So the fractions don't sum to 1 — that's fine. We're penalizing deviation from target representation *per color*, not a distribution over colors.

**Also penalize colorless:** Add a term for colorless cards (color identity = empty):

\[
P_{\text{color}}(C) = \sum_{c \in \{W, U, B, R, G\}} \left(\frac{n_c(C)}{|C|} - t_c\right)^2 + \left(\frac{n_{\emptyset}(C)}{|C|} - t_{\emptyset}\right)^2
\]

### Default targets (for a 360-card cube)

These are starting points to be tuned:

| Category | Target fraction | Target count (of 360) |
|---|---|---|
| White | 0.18 | ~65 |
| Blue | 0.18 | ~65 |
| Black | 0.18 | ~65 |
| Red | 0.18 | ~65 |
| Green | 0.18 | ~65 |
| Colorless | 0.10 | ~36 |

These overlap (multicolor cards count toward multiple colors), so the counts sum to more than 360. That's intentional.

### Data source

Color identity comes from `data/card_metadata.json`, field `color_identity` (a list like `["W", "U"]` or `[]` for colorless).

---

## Term 3: Type Balance Penalty

\[
P_{\text{type}}(C) = \left(\frac{n_{\text{creature}}(C)}{|C|} - t_{\text{creature}}\right)^2 + \left(\frac{n_{\text{noncreature\_nonland}}(C)}{|C|} - t_{\text{noncreature}}\right)^2 + \left(\frac{n_{\text{land}}(C)}{|C|} - t_{\text{land}}\right)^2
\]

Three-way split: creatures, noncreature spells, and lands. A card is classified by its `type_line` from Scryfall:
- **Creature:** type_line contains "Creature" (includes "Artifact Creature", "Enchantment Creature", etc.)
- **Land:** type_line contains "Land"
- **Noncreature nonland:** everything else (instants, sorceries, enchantments, artifacts, planeswalkers)

### Default targets

| Category | Target fraction | Target count (of 360) |
|---|---|---|
| Creature | 0.45 | ~162 |
| Noncreature spell | 0.45 | ~162 |
| Land | 0.10 | ~36 |

Creatures at roughly half is standard for a balanced cube. The land count is modest — these are utility/fixing lands (Sol Ring doesn't count as a land, but Command Tower does). Drafters add basic lands from outside the cube.

### Data source

`type_line` from `data/card_metadata.json`.

---

## Term 4: Mana Curve Penalty

\[
P_{\text{curve}}(C) = \sum_{b \in \text{CMC buckets}} \left(\frac{n_b(C)}{|C|} - t_b\right)^2
\]

Bucket cards by converted mana cost (CMC). Only count nonland cards (lands have CMC 0 but shouldn't inflate the 0-CMC bucket).

### CMC buckets and default targets

| Bucket | CMC range | Target fraction (of nonland cards) | Target count (of ~324 nonland) |
|---|---|---|---|
| 0-1 | CMC ≤ 1 | 0.15 | ~49 |
| 2 | CMC = 2 | 0.22 | ~71 |
| 3 | CMC = 3 | 0.22 | ~71 |
| 4 | CMC = 4 | 0.18 | ~58 |
| 5 | CMC = 5 | 0.12 | ~39 |
| 6+ | CMC ≥ 6 | 0.11 | ~36 |

These are rough targets for a "normal" curve that leans slightly aggressive. Tunable.

**Note:** The curve penalty fractions are out of nonland cards, not total cards. So `n_b(C)` is the count of nonland cards in bucket b, and the denominator is `|C| - n_land(C)`, not `|C|`.

### Data source

`cmc` from `data/card_metadata.json`. It's a float (e.g. 3.0). Floor to int for bucketing.

---

## Lambda Values

Three lambdas: `λ_color`, `λ_type`, `λ_curve`. Each controls how much its penalty matters relative to the synergy term.

**Scaling intuition:** The synergy term produces values in the range of roughly `k · ln(1 + avg_incoming)`. For k=360 and moderate average incoming synergy, this might be on the order of 500-2000 (rough guess — depends heavily on the data). The penalty terms are sums of squared fractions, so they're on the order of 0.01-0.1 for moderate imbalance. The lambdas need to bridge this scale gap.

**Starting values:** Set all three lambdas high enough that a badly imbalanced cube pays a noticeable penalty, but not so high that the optimizer ignores synergy entirely. This requires experimentation. A reasonable starting approach:

1. Run with all lambdas = 0 (pure synergy). Observe how imbalanced the result is.
2. Increase lambdas until the imbalance is tolerable.
3. Check that the synergy score hasn't collapsed.

Alternatively, normalize the penalty terms by dividing by the number of buckets (so each penalty is an *average* squared deviation), making the lambdas more comparable across terms.

---

## Solver: Greedy + Local Search

The log objective is not linearizable, so ILP is out. Greedy + local search is the intended solver.

### Algorithm

```
1. Initialize: Pick k cards randomly (or seeded from top-synergy cards).
2. Repeat until no improving swap found:
   a. For each card A in the cube:
      For each card B not in the cube:
        Compute delta = f(C - {A} + {B}) - f(C)
   b. Accept the swap (A_out, B_in) with the largest positive delta.
3. Return C.
```

### Efficient delta computation

When swapping A_out for B_in:

**Synergy term delta:** Removing A_out changes the incoming synergy for every card that A_out contributed to (i.e., every card X where S[A_out, X] > 0). Adding B_in changes the incoming synergy for every card that B_in contributes to. For each affected card X, recompute ln(1 + incoming(X)).

The number of affected cards is bounded by the number of nonzero entries in A_out's row and B_in's row of the synergy matrix. For sparse matrices, this is fast.

**Penalty term delta:** Just update the running counts (e.g., decrement white count if A_out is white, increment if B_in is white) and recompute the quadratic terms. O(1).

### Computational cost

Each swap evaluation: O(nnz_per_row) for synergy + O(1) for penalties.
Each iteration: O(k × n × nnz_per_row) to evaluate all possible swaps.
Total: O(iterations × k × n × nnz_per_row).

For k=360, n=5000, nnz_per_row ≈ 50 (guess), 100 iterations: 360 × 5000 × 50 × 100 = 9 billion operations. That's a few minutes. Could be accelerated by only considering a random subset of swaps per iteration, or by maintaining a priority queue of promising swaps.

---

## What v1 Does NOT Include

Explicitly left out, to keep the first version simple:

- **Popularity regularization** (α term) — add in v2 after we see if pure synergy picks weird cards.
- **Surprise weighting** (β term) — add later, requires TF-IDF preprocessing.
- **Color pair synergy balance** — add later, after basic color count balance is working.
- **Small-world bonus** — add later, computationally heavier.
- **Power level uniformity** — add later, need a good power proxy first.
- **Matrix symmetrization** — not needed. The incoming direction handles asymmetry directly.
- **ILP solver** — replaced by greedy + local search.

---

## Implementation Checklist

1. **Load data:** Synergy matrix (from `load_cards_files.py`), card metadata (from `data/card_metadata.json`).
2. **Build card property lookup:** For each card, store color_identity, cmc, type classification (creature/noncreature/land).
3. **Implement objective function:** `evaluate(cube) -> float` that computes the full f(C).
4. **Implement delta function:** `evaluate_swap(cube, card_out, card_in) -> float` that computes the change in f(C) efficiently.
5. **Implement greedy + local search:** The main optimization loop.
6. **Set default targets and lambdas.** Start with the values in this doc.
7. **Run and inspect.** Print the resulting cube, its synergy score, color distribution, type distribution, mana curve. Visually inspect for sanity.
8. **Tune lambdas.** Iterate on step 6-7 until the output looks like a real cube.
