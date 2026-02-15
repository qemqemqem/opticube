# Optimizer: Bugs, Fixes, and Redesign

This doc covers what's wrong with the current optimizer and the plan to fix and improve it. It assumes the data pipeline (scraper rewrite, Scryfall download, matrix construction) is handled separately -- see `DATA_LOGISTICS.md`.

---

## Known Bugs

### 1. Asymmetry Bug (correctness)

The ILP iterates over non-zero entries with `i <= j` and uses `S[i,j]` as the weight. But the synergy matrix is asymmetric -- `S[i,j]` and `S[j,i]` are different values from different EDHREC pages. The current code arbitrarily picks whichever direction has `i <= j`, discarding the other.

**Fix:** Symmetrize the matrix before solving.

```python
S_sym = (S + S.T) / 2
```

For entries where only one direction exists, this halves the value (since the other direction is zero). An alternative: use `S + S.T` without dividing, which preserves magnitude but double-counts pairs where both directions exist. The right choice depends on how we want to treat one-sided vs two-sided evidence. The safest option:

```python
# Where both directions exist, average them.
# Where only one exists, use it as-is.
S_both = S.multiply(S.T > 0)           # entries where both directions exist
S_one = S.multiply(S.T == 0)           # entries where only one direction exists
S_sym = (S_both + S_both.T) / 2 + S_one + S_one.T
```

### 2. Pre-filter Loses the Optimal Solution (correctness)

`reduce_matrix` keeps the top `k * 10` cards by `row_sum + col_sum`. This is a global popularity metric. The optimal synergistic subset might not contain the globally most popular cards -- it might be a tight cluster of cards with high mutual synergy but low total connectivity.

**Fix (short term):** Increase the multiplier from 10 to something larger (50-100), and make it configurable.

**Fix (long term):** Replace the pre-filter with a smarter candidate selection. See "Spectral Pre-filtering" below.

### 3. Card Name Indexing Bug After Reduce

After `reduce_matrix`, the returned matrix has new indices (0 to `top_n - 1`), but `main()` uses the solution indices to look up `card_names[i]` as if they're indices into the original full matrix. This means the printed card names are **wrong** -- they're the names of cards 0 through 9 in the original list, not the actual selected cards.

**Fix:** `reduce_matrix` needs to return the mapping from new indices to original indices:

```python
def reduce_matrix(matrix, top_n):
    total_sums = row_sums + col_sums
    top_indices = np.argsort(total_sums)[-top_n:]
    reduced_matrix = matrix[np.ix_(top_indices, top_indices)]
    return reduced_matrix, top_indices  # return the mapping

# In main():
reduced_matrix, index_map = reduce_matrix(synergy_matrix, top_n)
solution = find_good_subset_ilp(reduced_matrix, set_size=10)
original_indices = [index_map[i] for i in solution]
names = [card_names[i] for i in original_indices]
```

### 4. Solver Settings Are Too Aggressive

- `timeLimit=10` -- 10 seconds is not enough for a serious ILP.
- `gapRel=0.05` -- 5% gap means the answer could be 5% below optimal.
- `maxSolutions 1` -- stops after the *first feasible* solution, which may be far from optimal.

**Fix:** Remove `maxSolutions 1`. Increase time limit to 60-300 seconds. Keep the gap tolerance at 5% (that's reasonable). Let the solver actually search.

---

## Redesigning the Objective Function

The current objective -- maximize total pairwise synergy -- finds the tightest cluster of mutually synergistic cards. This is wrong for cube design. A good cube needs **diversity**: multiple archetypes, color balance, a mana curve, and a mix of card types.

### Phase 1: Add Hard Constraints

These are requirements the solution must satisfy, encoded as linear constraints in the ILP.

**Color balance:** For a 360-card cube, require roughly equal representation across colors. Example:

```
60 <= |{i in C : W in color(i)}| <= 80    (white cards)
60 <= |{i in C : U in color(i)}| <= 80    (blue cards)
...
20 <= |{i in C : colorless(i)}| <= 40     (colorless cards)
```

Multi-color cards count toward each of their colors. These are linear constraints on the `x_i` variables (just filter by color and sum).

**Mana curve:** Require a distribution of CMCs:

```
|{i in C : cmc(i) <= 2}| >= 100
|{i in C : cmc(i) >= 6}| <= 40
```

**Card type mix:** Require minimum counts of creatures, instants, sorceries, lands, etc.

All of these are linear in the `x_i` variables and add no new auxiliary variables. They're cheap to add to the ILP.

### Phase 2: Archetype Diversity (the Hard Part)

The deeper problem: maximizing total synergy rewards a monoculture. If red burn spells are all synergistic with each other, the optimizer will pick 360 red burn spells. The hard constraints above prevent the worst cases, but they don't ensure the cube has *playable archetypes*.

**Approach A: Cluster-then-optimize**

1. Run spectral clustering or community detection on the synergy graph to identify natural archetypes (e.g., "spellslinger", "aristocrats", "ramp", "+1/+1 counters").
2. Require the solution to include a minimum number of cards from each cluster.
3. Maximize synergy subject to these cluster quotas.

This decomposes the problem: first find the archetypes, then fill each one optimally.

**Approach B: Multi-objective optimization**

Define two objectives:
- **Intra-archetype synergy:** Cards within the same archetype should be synergistic.
- **Inter-archetype balance:** The cube should have roughly equal support for each archetype.

This could be formulated as maximizing the minimum per-archetype synergy (a maximin objective), which can be linearized.

**Approach C: Keyword/theme quotas (simpler, pragmatic)**

Instead of discovering archetypes, manually define themes based on keywords and oracle text patterns:
- "+1/+1 counter" theme: cards mentioning "+1/+1 counter" in oracle text
- Lifegain theme: cards mentioning "gain" and "life"
- Graveyard theme: cards mentioning "graveyard"
- Tokens theme: cards mentioning "create" and "token"
- Spellslinger theme: instants/sorceries with "whenever you cast"

Require minimum representation (e.g., at least 15-20 cards per theme). This is crude but easy to implement as linear constraints using Scryfall oracle text data.

---

## Scaling the Solver

### Current: ILP with CBC

Works for k=10, pool=100. Won't scale to k=360, pool=5000.

### Option 1: Better ILP Solver

**Gurobi** (free academic license) handles much larger ILPs than CBC. It might push the boundary to k=360 with a pool of 1000-2000. The formulation stays the same.

### Option 2: LP Relaxation + Rounding

Relax `x_i` and `y_ij` from binary to continuous `[0, 1]`. Solve the LP (fast, polynomial time). Then round the fractional solution to binary. This gives no optimality guarantee, but LP relaxations of dense subgraph problems often have small integrality gaps in practice.

### Option 3: Greedy + Local Search (most practical for large k)

1. **Seed:** Start with one card per archetype/cluster (using spectral clustering).
2. **Greedy expansion:** Repeatedly add the card that increases total synergy the most, subject to constraints, until we reach k cards.
3. **Local search:** Repeatedly try swapping one card in the solution for one card outside. Accept the swap if it improves the objective. Repeat until no improving swap exists.

This is the approach most likely to scale to k=360 from a pool of 5,000+. It's fast (each step is O(n)), produces good solutions in practice, and naturally respects constraints (just check them before accepting a swap).

The greedy step is O(k * n) and the local search is O(iterations * k * n). For k=360, n=5000, and 1000 iterations, that's ~1.8 billion operations -- a few minutes on a modern machine.

### Option 4: Spectral Methods

The synergy matrix has eigenstructure. The leading eigenvectors represent the dominant synergy patterns (archetypes). Projecting cards into this low-dimensional space and clustering them could:
- Identify natural archetypes
- Provide good initial seeds for greedy/local search
- Guide the pre-filtering step (instead of "top by row sum", select diverse cards that span the principal components)

This is a preprocessing step, not a solver, but it makes everything downstream work better.

---

## Recommended Implementation Order

### Immediate Fixes (do these first, they're bugs)

1. **Fix the index mapping bug** in `reduce_matrix` / `main()` -- the results are currently wrong.
2. **Symmetrize the matrix** before solving.
3. **Remove `maxSolutions 1`** and increase the time limit.

### Short-term Improvements (after data pipeline is rebuilt)

4. **Add color balance constraints** using Scryfall color identity data.
5. **Add mana curve constraints** using Scryfall CMC data.
6. **Add card type constraints** (minimum creatures, minimum instants/sorceries, etc.).
7. **Increase the pre-filter multiplier** from 10x to 50-100x.
8. **Make `set_size` configurable** (command-line argument or config file).

### Medium-term Redesign

9. **Spectral analysis** of the synergy matrix to identify archetype clusters.
10. **Implement greedy + local search** as an alternative to ILP for larger problem sizes.
11. **Keyword/theme detection** from Scryfall oracle text for archetype quotas.
12. **Evaluate at k=30-50** as a stepping stone toward full cube size.

### Long-term Goals

13. **Scale to k=360** for a real cube, probably using greedy + local search with spectral seeding.
14. **Archetype-aware objective** (maximize per-archetype synergy, not just global synergy).
15. **Interactive mode** where you can lock in certain cards and optimize around them.
16. **Evaluation metrics** beyond raw synergy score: archetype coverage, draft simulation, mana curve analysis.
