# Objective Function: Refined Candidates

Winnowed from the brainstorm. These are the ideas that survived scrutiny, focused on what feels right for this project: continuous summation metrics, cards as the basic unit (no hand-defined archetypes), computationally tractable, theoretically clean.

---

## The Core: Incoming Log-Synergy

The central idea that keeps coming back:

\[
f(C) = \sum_{A \in C} \log\!\left(1 + \sum_{B \in C,\; B \neq A} \text{ReLU}\!\left(S[B, A]\right)\right)
\]

Unpack this piece by piece.

**Incoming synergy.** For card A, we sum up how much the *other* cards in the cube value A. That's `S[B, A]` -- the synergy of A as measured from B's perspective (B's EDHREC page). This is the column direction: "how wanted is card A by the rest of the cube?"

This resolves the asymmetry question cleanly. We don't need to symmetrize the matrix. Card A's score is determined by how much the other cards want A around, which is the natural direction for "does this card belong here?"

**ReLU.** Clip negative synergy to zero. Anti-synergy is normal in MTG -- a board wipe and a token maker are anti-synergistic but both are good cube cards. We don't want to penalize cards for having some enemies. We just want to reward cards for having friends. ReLU(x) = max(0, x).

**Log.** Concavity creates diminishing returns. Suppose card A has incoming synergy of 100 and card B has incoming synergy of 10. Under linear summation, A contributes 10x as much as B. Under log, A contributes log(101)/log(11) ≈ 1.9x as much. The log dramatically compresses the range.

This has a crucial structural consequence: **the optimizer prefers spreading synergy across many cards over concentrating it on a few.** Two cards with incoming synergy 50 each contribute 2 · log(51) ≈ 7.87. One card with 100 and one with 0 contribute log(101) + log(1) ≈ 4.62. The balanced case wins by a wide margin.

This is the anti-monoculture mechanism. The log naturally prevents the optimizer from piling everything into one dense archetype, because the marginal value of "yet another burn spell that likes burn spells" decreases as you add more of them.

**The "budget" intuition.** The user's insight: "if a card is part of an archetype, then it has limited value to spread around." The log creates this effect. In a dense archetype of 50 mutually synergistic cards, each card has high incoming synergy, but the log flattens it all. Meanwhile, a bridge card that connects two sparse archetypes gets disproportionate reward -- its incoming synergy might be modest, but it comes from cards that have few other synergy partners (and thus each contribution matters more on the margin of the *contributor's* log term).

Wait -- that last point is about the contributor's side, not the receiver's. This is actually subtle. The log operates on the receiver. The contributor's log is affected by how many other cards value *them*. So the metric doesn't directly enforce the "budget" property on the contribution side. But it does enforce it on the receiving side, which achieves something similar: every card needs to be wanted, and being *very* wanted has diminishing returns.

---

## Variance-Penalized Synergy (and where it comes from)

\[
f(C) = \frac{1}{k}\sum_{A \in C} v_A \;-\; \lambda \cdot \frac{1}{k}\sum_{A \in C} (v_A - \bar{v})^2
\]

where \( v_A = \sum_{B \in C} S[B, A] \) is card A's incoming synergy and \( \bar{v} \) is the mean across all cards in the cube.

**Origin: Modern Portfolio Theory (Markowitz, 1952).** In finance, the core insight is that you shouldn't just maximize expected return -- you should maximize expected return *minus a penalty for risk (variance)*. A portfolio of assets that are individually mediocre but uncorrelated beats a portfolio of one high-return, high-variance asset.

The analogy to cube design: each card is an "asset." Its "return" is how much synergy it brings to the cube. Its "risk" is how uneven that contribution is relative to the other cards. A cube where one card has enormous synergy and most have none is "high variance" -- it's fragile. Lambda (λ) controls the tradeoff: λ = 0 is pure synergy maximization, large λ prioritizes balance.

**Relationship to log-synergy:** Log-synergy and variance-penalized synergy are *different* mechanisms that achieve *similar* goals. The log version is simpler (one formula, no λ parameter) and the concavity is "built in." The variance version is more explicit about the tradeoff and gives you a knob to turn.

In practice, log-synergy subsumes much of the benefit of variance penalization. The log *implicitly* penalizes variance because it's concave. You could combine them, but the log alone may be sufficient.

**Verdict:** The log approach is cleaner. Variance penalization is the theoretical backing for *why* the log works -- it's the intellectual ancestor, not a separate thing to bolt on.

---

## Popularity Regularization

All else equal, prefer cards that are proven playable. A card that appears in 50,000 Commander decks is more likely to be a real, fun card than one that appears in 12. We don't want the optimizer chasing obscure synergy between niche cards that nobody actually plays.

\[
f(C) = \sum_{A \in C} \left[\log\!\left(1 + \text{incoming}(A)\right) + \alpha \cdot \log\!\left(1 + \text{popularity}(A)\right)\right]
\]

where `popularity(A)` could be:
- Number of Commander decks containing A (from EDHREC)
- Number of cubes containing A (from CubeCobra, if we get that data)
- Log of number of decks (double-logging, which is aggressive dampening)

The popularity term acts as a **prior** or **regularizer**. Without it, the optimizer might find a technically synergistic subset of obscure cards nobody has heard of. With it, the optimizer is nudged toward cards that real players actually like.

Alpha (α) controls the strength of this prior. α = 0 means pure synergy. Large α means "just pick popular cards." Something in between means "find the most synergistic subset among reasonably popular cards."

**What about using cubes instead of decks?** Deck co-occurrence is a stronger signal of "these cards work together in a single game." Cube co-occurrence is a different signal: "these cards belong in the same drafting environment," which is arguably more relevant. But deck co-occurrence is what we have from EDHREC, and it's a reasonable starting point. CubeCobra data could be a future upgrade.

---

## Surprise Bonus

Reward synergy between cards that *don't obviously go together*. Two cards that share keywords ("whenever you gain life" + "you gain 3 life") are obviously synergistic -- a human designer would find that pairing without data. The interesting pairs are cards that synergize for non-obvious reasons.

\[
\text{incoming\_surprise}(A) = \sum_{B \in C} \text{ReLU}(S[B, A]) \cdot (1 - \text{text\_sim}(A, B))
\]

where `text_sim(A, B)` is the cosine similarity of TF-IDF vectors computed from oracle text. High synergy between textually dissimilar cards gets full weight. High synergy between textually similar cards is discounted.

This would make the optimizer prefer cubes with more "discoverable" interactions -- pairs where drafters go "wait, these cards are great together?" rather than "obviously lifegain card goes with lifegain card."

**Computational note:** TF-IDF vectors for all cards can be precomputed once from Scryfall oracle text. The text_sim matrix is dense but only needs to be computed for card pairs that have nonzero synergy (which is sparse). Feasible.

**Tuning concern:** This is a multiplier on synergy, so it reshapes the synergy landscape rather than adding a new term. Setting text_sim = 0 for all pairs recovers the base objective. Could interpolate: `weight = 1 - β · text_sim(A, B)` where β ∈ [0, 1] controls how much we discount obvious synergy.

---

## Color Balance via Soft Penalty

Hard constraints ("exactly 60 white cards") are brittle. Lagrangian relaxation turns them into soft penalties:

\[
\text{penalty}_{\text{color}} = \lambda_{\text{color}} \sum_{c \in \text{colors}} \left(\frac{|\{A \in C : c \in \text{colors}(A)\}|}{|C|} - \text{target}_c\right)^2
\]

This is a sum of squared deviations from target color proportions. The quadratic shape means small deviations are cheap and large deviations are very expensive. Lambda controls how much we care.

Same approach works for:
- **Mana curve:** Penalize deviation from a target CMC distribution.
- **Card types:** Penalize too few creatures, too few instants, etc.
- **Lands:** Penalize deviation from target land count.
- **Power level uniformity:** Penalize high variance in card power level (estimated from inclusion rate or some other proxy).

Each gets its own lambda. The full objective becomes:

\[
f(C) = \underbrace{\sum_A \log(1 + \text{incoming}(A))}_{\text{synergy}} + \underbrace{\alpha \sum_A \log(1 + \text{pop}(A))}_{\text{popularity}} - \underbrace{\lambda_1 \cdot \text{color\_penalty}}_{\text{balance}} - \underbrace{\lambda_2 \cdot \text{curve\_penalty}}_{\text{curve}} - \;\cdots
\]

**Why Lagrangian over hard constraints?** Two reasons. First, the solver we'll likely use (greedy + local search) handles soft penalties more naturally than hard constraints -- every candidate swap just changes the objective by a computable delta. Second, soft penalties allow graceful tradeoffs. Maybe the absolute best synergy requires 70 red cards and 40 blue cards. A hard constraint forces exactly 60/60. A soft penalty lets the optimizer decide whether the synergy gain is worth the color imbalance.

**Why Lagrangian over min/max?** Continuous penalties compose better. Min-based objectives create flat regions in the objective landscape (if the worst color pair has synergy 20 and you improve the second-worst from 25 to 30, the objective doesn't change). Quadratic penalties always respond to improvements, giving the optimizer a smooth gradient to follow.

---

## Color Pair Synergy Balance

Beyond raw color count balance, we want every two-color pair to be a viable draft archetype. Define the color pair synergy:

\[
\text{pair\_synergy}(c_1, c_2) = \sum_{\substack{A, B \in C \\ c_1 \in \text{colors}(A) \\ c_2 \in \text{colors}(B)}} \text{ReLU}(S[B, A])
\]

Then penalize imbalance across the 10 color pairs:

\[
\text{penalty}_{\text{pairs}} = \lambda_{\text{pairs}} \cdot \text{Var}\!\left(\text{pair\_synergy}(c_1, c_2) \;\text{for all 10 pairs}\right)
\]

Minimizing variance across color pairs means no color combination is dramatically better or worse than any other for drafting. This is a soft, continuous penalty -- consistent with the Lagrangian philosophy.

**Alternative: normalize each pair's synergy by the number of cards in those colors**, so a color pair with fewer cards isn't automatically penalized. The variance would then be over "synergy density per color pair."

---

## The Small-World Property

A "small-world" graph has two properties simultaneously: **high clustering** (friends of your friends are your friends) and **short average path length** (any card is reachable from any other card in a few hops). This is the ideal cube structure:

- High clustering = archetypes exist (tight groups of mutually synergistic cards).
- Short path length = archetypes are connected (you can pivot between strategies mid-draft).

Measuring small-world-ness on the induced synergy subgraph:

\[
\text{SW}(C) = \frac{\text{CC}(C) / \text{CC}_{\text{random}}}{\text{APL}(C) / \text{APL}_{\text{random}}}
\]

where CC is the clustering coefficient, APL is average path length, and the "random" subscript refers to an Erdos-Renyi random graph with the same number of nodes and edges.

SW >> 1 means the graph is "more small-world than random." This is a single scalar that captures both archetype structure (clustering) and archetype connectivity (path length).

**As a term in the objective:** Add `+ γ · SW(C)` to reward small-world structure. Gamma controls how much we care. Computing CC and APL is O(n^3) in the worst case but feasible for n = 360 with optimized algorithms (or approximations).

**Parameters:** Gamma is the one knob. But the definition of "edge" in the synergy graph also matters -- do we threshold (synergy > t means an edge exists) or use weighted graph metrics? Thresholding introduces a second parameter (t). Weighted versions exist for both CC and APL.

---

## Choice of Base Measure

EDHREC "synergy" is one way to measure card association. Others:

| Measure | Formula | Symmetric? | Notes |
|---|---|---|---|
| EDHREC synergy | P(B\|A) - P(B) | No | What we have. Measures "excess co-occurrence." |
| Lift | P(B\|A) / P(B) | No* | Already in EDHREC data. Ratio version of synergy. |
| PMI | log(P(A,B) / P(A)P(B)) | Yes | = log(lift). Information-theoretically principled. |
| NPMI | PMI / -log(P(A,B)) | Yes | Normalized to [-1, 1]. Comparable across frequencies. |
| Jaccard | \|both\| / \|either\| | Yes | Doesn't separate synergy from popularity. |
| Confidence-weighted | S[B,A] · (1 - 1/sqrt(n_decks)) | No | Down-weights low-sample-size synergy. |

*Lift is symmetric in the population sense but asymmetric in the EDHREC data because each direction comes from a different page with different sample sizes.

**PMI is the most principled.** It's symmetric, grounded in information theory, and naturally handles the base rate issue (popular cards don't get inflated scores just for being popular). We can compute it from EDHREC data: the `inclusion` and `potential_decks` fields give us P(B|A) = inclusion/potential_decks, and we can estimate P(B) from overall inclusion rates.

**Confidence weighting matters.** A +50% synergy between two cards that appear in 15 decks together is noise. The same synergy across 15,000 decks is signal. Whatever base measure we choose, weighting by confidence (sample size) is important.

**Deck vs. cube co-occurrence.** Deck co-occurrence (EDHREC) measures "these cards work together in a game." Cube co-occurrence (CubeCobra) measures "these cards belong in the same draft environment." The latter is arguably more relevant -- two cards in the same cube might be in different archetypes and never appear in the same deck, but they both make the draft better. Deck co-occurrence is a stronger signal per observation but measures a different thing. Both could be useful; deck co-occurrence is what we have now.

---

## Computational Reality

**The log-synergy objective is not linearizable.** This means we can't use ILP (the current approach). But that's fine -- ILP doesn't scale to k=360 anyway. The intended solver for full-size cubes is **greedy + local search**, which can evaluate *any* objective function, including logs, ReLUs, quadratic penalties, and graph metrics.

**Greedy + local search works like this:**
1. Start with a random feasible cube (or seeded from some heuristic).
2. For each possible swap (remove card A, add card B): compute Δf = f(C - A + B) - f(C).
3. Accept the best improving swap. Repeat until no improving swap exists.

Each swap evaluation is O(k) for the log-synergy term (recompute the incoming synergy for the affected cards). The penalty terms are O(1) per swap (just update the running counts). Total: O(iterations × k × n), which is fast for k=360, n=5000.

**The surprise bonus** requires precomputing TF-IDF similarity, which is a one-time O(n²) cost (or O(nnz) if we only compute it for synergy pairs). Then it's just a multiplication during evaluation.

**The small-world metric** is more expensive (clustering coefficient is O(k³) naively). But for k=360 it's ~47 million operations -- a few seconds. And we don't need to recompute it from scratch on every swap; local updates are possible.

---

## Putting It Together: A Candidate Objective

\[
f(C) = \sum_{A \in C} \log\!\left(1 + \sum_{B \in C, B \neq A} w_{BA}\right) + \alpha \sum_{A \in C} \log(1 + \text{pop}(A)) - \lambda_{\text{color}} \cdot P_{\text{color}}(C) - \lambda_{\text{curve}} \cdot P_{\text{curve}}(C) - \lambda_{\text{pairs}} \cdot P_{\text{pairs}}(C)
\]

where:
- \( w_{BA} = \text{ReLU}(S[B, A]) \cdot (1 - \beta \cdot \text{text\_sim}(A, B)) \) — surprise-weighted incoming synergy
- \( \text{pop}(A) \) — popularity of card A (deck count, cube count, or similar)
- \( P_{\text{color}}(C) \) — squared deviation from target color proportions
- \( P_{\text{curve}}(C) \) — squared deviation from target mana curve
- \( P_{\text{pairs}}(C) \) — variance in synergy across the 10 color pairs

Parameters: α, β, λ_color, λ_curve, λ_pairs. Five knobs total. Could add more (land count, power level, small-world bonus), but five is a reasonable starting point.

**For a first implementation**, simplify: set β = 0 (no surprise weighting), drop the color pair variance term, and just use:

\[
f(C) = \sum_{A \in C} \log\!\left(1 + \sum_{B \in C, B \neq A} \text{ReLU}(S[B, A])\right) - \lambda \sum_c \left(\text{color\_frac}_c - \text{target}_c\right)^2
\]

Two terms, one parameter (λ). That's a v1 we can build and evaluate.

---

## Open Questions (smaller list now)

1. **Log base.** Natural log? Log base 10? Doesn't matter for optimization (they're proportional), but affects the relative scale of the synergy term vs. the penalty terms, and thus the "natural" range of λ values. Probably just use ln and tune λ empirically.

2. **What counts as "incoming synergy = 0"?** If a card has zero incoming synergy from everything in the cube, log(1 + 0) = 0. That card contributes nothing. Is that okay, or should we explicitly penalize zero-synergy cards? The current formulation already handles this -- zero-synergy cards are dead weight that could be swapped for something better.

3. **Should the popularity term use EDHREC data or CubeCobra data?** EDHREC gives deck count (Commander format). CubeCobra gives cube inclusion count (draft format). CubeCobra is more directly relevant but requires a new data source. Start with EDHREC, upgrade later.

4. **How to set λ values?** Run the optimizer with several values and look at the output cubes. Too low λ = color imbalance. Too high λ = boring cube that's perfectly balanced but has no synergy. The interesting region is in between. Could also use the constraint-tightening approach: start with λ = 0 and gradually increase until the cube "looks right."

5. **Is the incoming direction correct?** We chose "how much do other cards value A" over "how much does A value other cards." Intuition says the incoming direction is right (a card should be *wanted*). But worth validating empirically -- do the two directions give meaningfully different results?
