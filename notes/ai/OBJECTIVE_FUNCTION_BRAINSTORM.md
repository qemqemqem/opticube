# Objective Function Brainstorm

What should the optimizer actually be maximizing? This doc explores a wide range of ideas without committing to any single one. The goal is to map the design space before narrowing down.

---

## The Fundamental Question

A cube is a set of cards designed for drafting. "Good" is subjective and multidimensional. The current formulation -- maximize total pairwise synergy -- captures one narrow slice of quality. What else could we optimize?

Before listing approaches, it helps to ask: **what does a cube designer actually want?**

- Every drafter can build a coherent deck
- Multiple distinct strategies are viable
- No single strategy dominates
- Draft decisions feel meaningful (not "pick the best card in a vacuum")
- Cards interact in satisfying, discoverable ways
- The cube is replayable -- different drafts feel different
- Color balance, mana curve, card type mix are reasonable

These are all different things, and no single objective captures all of them.

---

## Category 1: Synergy Aggregation Variants

These all use the same pairwise synergy data but aggregate it differently.

### 1.1 Total Pairwise Synergy (current)

\[
f(C) = \sum_{i \in C} \sum_{j \in C} S[i,j]
\]

Finds the densest cluster. Problem: monoculture. A cube of 360 red burn spells would score well if they're all synergistic.

### 1.2 Log-Synergy (diminishing returns)

\[
f(C) = \sum_{i \in C} \log\left(1 + \sum_{j \in C} S[i,j]\right)
\]

Each card's total synergy contribution has diminishing returns. This naturally penalizes cards with one massive synergy partner but nothing else, and rewards cards that have moderate synergy with *many* partners. The log is concave, so adding another burn spell to a cube already full of burn gives less marginal benefit than adding a card that bridges two themes.

### 1.3 Thresholded Synergy (binary edges)

Pick a threshold `t`. Define:

\[
f(C) = |\{(i,j) \in C \times C : S[i,j] > t\}|
\]

Count the number of "strong synergy" pairs rather than summing raw values. This is equivalent to finding a dense subgraph in an unweighted graph. Less sensitive to outlier synergy values. The choice of `t` becomes a parameter to tune.

### 1.4 Top-K Synergy Per Card

For each card `i`, only count its `K` strongest synergy partners within the cube:

\[
f(C) = \sum_{i \in C} \sum_{j \in \text{top-K}_C(i)} S[i,j]
\]

This ensures every card has *some* good partners without requiring it to synergize with the entire cube. K could be 5-15 (representing "the card should work well in at least a few decks"). This directly models the intuition that a cube card doesn't need to be good with everything -- it needs to be *great* with a handful of things.

Downside: harder to optimize because the "top K" set changes as the solution changes. Would likely need a heuristic solver (greedy/local search) rather than ILP.

### 1.5 Minimum Card Synergy (Maximin)

\[
f(C) = \min_{i \in C} \sum_{j \in C} S[i,j]
\]

Maximize the synergy of the *worst* card in the cube. No card is a dead-weight filler. This is linearizable (introduce a variable `z` and add constraints `z <= sum_j S[i,j] * x_j` for all `i`, then maximize `z`). It's a classic maximin formulation.

Problem: this could be too conservative. It might include many mediocre cards instead of a few excellent ones plus a few role-players.

### 1.6 Percentile Synergy

Instead of the minimum, use the 10th or 25th percentile card synergy. This is more robust than pure maximin -- allows a few low-synergy cards (staples, mana fixing) while ensuring most cards pull their weight.

Harder to linearize directly. Could approximate by requiring that at least `0.9 * k` cards have synergy above some threshold.

### 1.7 Geometric Mean Synergy

\[
f(C) = \prod_{i \in C} \left(\sum_{j \in C} S[i,j]\right)^{1/k}
\]

Equivalent to maximizing the sum of log-synergies. Strictly penalizes any card with zero synergy (the product goes to zero). Encourages a balanced distribution of synergy across all cards.

### 1.8 Variance-Penalized Synergy

\[
f(C) = \text{mean}_{i \in C}\left(\sum_j S[i,j]\right) - \lambda \cdot \text{var}_{i \in C}\left(\sum_j S[i,j]\right)
\]

Like portfolio theory -- maximize expected synergy while penalizing uneven distribution. Lambda controls the tradeoff. Higher lambda means more balanced synergy across cards.

### 1.9 Negative Synergy as Signal

The current approach ignores negative synergy (or treats it as noise). But negative synergy is information! If two cards actively anti-synergize (e.g., "destroy all creatures" and a creature-heavy theme), that's important.

\[
f(C) = \sum_{i,j \in C} S^+[i,j] - \alpha \sum_{i,j \in C} S^-[i,j]
\]

where `S+` and `S-` are the positive and negative parts. Alpha controls how much we penalize anti-synergy. Setting alpha > 1 means we care more about avoiding bad pairings than we do about adding good ones.

---

## Category 2: Graph-Theoretic Objectives

The synergy matrix defines a weighted graph. Many graph properties are relevant to cube quality.

### 2.1 Minimum Degree

\[
f(C) = \min_{i \in C} \deg_C(i)
\]

where `deg_C(i)` is the number of cards in `C` that card `i` has positive synergy with. Ensures every card has a minimum number of synergy partners. A card with degree 0 in the cube is a dead card.

### 2.2 Algebraic Connectivity (Fiedler Value)

The second-smallest eigenvalue of the Laplacian of the induced subgraph. High algebraic connectivity means the graph is hard to disconnect -- there's no single "bridge" whose removal splits the cube into isolated halves. This is a mathematical proxy for "the cube is cohesive."

Not easy to optimize directly, but useful as an evaluation metric or constraint.

### 2.3 Clustering Coefficient

\[
\text{CC}(C) = \frac{1}{|C|} \sum_{i \in C} \frac{|\{(j,k) : j,k \in N_C(i),\; S[j,k] > 0\}|}{|N_C(i)|(|N_C(i)|-1)/2}
\]

High clustering coefficient means synergy partners of a card also tend to synergize with each other -- "friends of friends are friends." This is the structure that creates discoverable combos and cohesive archetypes.

### 2.4 Community Structure (Modularity)

Newman's modularity measures how well a graph decomposes into clusters:

\[
Q = \frac{1}{2m}\sum_{ij}\left[S[i,j] - \frac{d_i d_j}{2m}\right]\delta(c_i, c_j)
\]

A cube with high modularity has clear archetypes (tight clusters) but still has some connections between them. This is subtler than just "maximize synergy" -- it rewards *structured* synergy over undifferentiated blob synergy.

Could use modularity as a constraint: require `Q >= threshold` to ensure the cube has identifiable archetypes.

### 2.5 Community Coverage

Run community detection on the full card graph first. Then require the cube to include cards from at least `m` communities, with minimum representation per community. This doesn't optimize for synergy directly -- it optimizes for thematic diversity.

### 2.6 Bridge Cards (Betweenness Centrality)

Some cards bridge between archetypes -- they're good in multiple strategies. These are valuable in drafts because they keep options open. We could add a bonus for cards with high betweenness centrality in the synergy graph:

\[
f(C) = \text{synergy}(C) + \beta \sum_{i \in C} \text{betweenness}(i)
\]

### 2.7 Small World Property

A "small world" graph has high clustering (local structure) but short average path length (global connectivity). This might be the ideal cube structure: tight archetypes (clustering) but enough bridge cards that you can pivot strategies mid-draft (short paths).

---

## Category 3: Coverage and Diversity

These approaches focus on ensuring the cube has breadth, not just depth.

### 3.1 Submodular Coverage

Define a "coverage function" where each card "covers" a set of themes, mechanics, or synergy partners. A submodular function has diminishing returns -- the 5th lifegain card adds less marginal value than the 1st.

\[
f(C) = \sum_{t \in \text{themes}} \min\left(\sum_{i \in C} w_{it}, \; \text{cap}_t\right)
\]

where `w_it` is how much card `i` contributes to theme `t`, and `cap_t` is a saturation point. Once a theme is "full," additional cards for that theme don't help.

This is a *weighted coverage* function. It's submodular, which means the greedy algorithm gives a (1 - 1/e) approximation. Very practical for large instances.

### 3.2 Determinantal Point Process (DPP)

A DPP is a probability distribution over subsets that naturally balances quality and diversity. The probability of selecting subset `C` is proportional to:

\[
\det(L_C)
\]

where `L` is a kernel matrix and `L_C` is the submatrix indexed by `C`. If `L = Q \cdot D \cdot Q^T` where `Q` encodes features and `D` encodes quality, the DPP favors diverse, high-quality subsets.

This is theoretically elegant but might be overkill. MAP inference (finding the highest-probability set) is NP-hard in general, but good greedy approximations exist.

### 3.3 Facility Location

Treat cube construction as a facility location problem: each selected card is a "facility" that "serves" nearby cards. We want every card in the universe to be close to at least one selected card:

\[
\min_{C} \max_{j \notin C} \min_{i \in C} d(i, j)
\]

where `d(i,j) = 1 / (1 + S[i,j])` or similar. This ensures the cube *represents* the full space of possible strategies, not just one corner of it.

### 3.4 Max-Min Color Pair Synergy

For each of the 10 two-color pairs (WU, WB, WR, WG, UB, UR, UG, BR, BG, RG), compute the total synergy among cards in the cube that belong to that color pair. Then maximize the minimum across all 10:

\[
f(C) = \min_{\text{pair} \in \text{color pairs}} \text{synergy}_{\text{pair}}(C)
\]

This directly encodes "every color pair should be a viable draft archetype." It's a maximin problem, which is linearizable.

### 3.5 Keyword/Theme Entropy

Define a set of mechanical themes (lifegain, tokens, graveyard, artifacts, etc.). For each card, compute its theme membership vector. Then measure the Shannon entropy of the theme distribution in the cube:

\[
H(\text{themes}) = -\sum_{t} p_t \log p_t
\]

where `p_t` is the fraction of theme-weight devoted to theme `t`. Maximizing entropy = maximizing mechanical diversity. This has no relation to synergy at all -- it's purely about breadth.

---

## Category 4: Draft Simulation Objectives

These treat the draft itself as the unit of evaluation.

### 4.1 Average Drafted Deck Quality

Simulate `N` drafts (e.g., 8 players, 3 packs of 15 cards each = 360 cards). Each AI drafter builds the best deck they can from their picks. Evaluate each deck by its internal synergy. The objective is the average deck quality across all players and simulations:

\[
f(C) = \mathbb{E}_{\text{drafts}}\left[\frac{1}{8}\sum_{p=1}^{8} \text{quality}(\text{deck}_p)\right]
\]

This is the gold standard -- it directly measures what we care about. But it's expensive (requires a draft simulator and deck-building AI) and hard to differentiate or use in a gradient-based optimizer. Would need to be evaluated via sampling.

### 4.2 Draft Branching Factor

At each pick in a simulated draft, count how many "good" options the drafter has (cards that improve their deck by at least some threshold). The branching factor measures how many viable strategies are available at each decision point:

\[
f(C) = \mathbb{E}\left[\frac{1}{\text{picks}} \sum_{\text{pick}} |\text{good options at pick}|\right]
\]

High branching factor = the draft has interesting decisions. Low branching = the draft is on rails (there's always one obviously correct pick).

### 4.3 Drafter Equity

After simulating drafts, measure the variance of deck quality across the 8 drafters. The cube is balanced if all drafters end up with roughly equally strong decks:

\[
f(C) = -\text{Var}(\text{deck quality across players})
\]

(Minimize variance, or equivalently maximize the negative of variance.)

### 4.4 Archetype Availability

In each simulated draft, count how many distinct archetypes are represented among the 8 drafted decks. A good cube supports 8+ distinct strategies so drafters aren't fighting over the same cards:

\[
f(C) = \mathbb{E}[\text{number of distinct archetypes drafted}]
\]

### 4.5 Signal Clarity

In a real draft, players read "signals" -- if red cards are flowing (not being picked), that signals red is open. A cube has good signals if early picks are informative about what's available. This is related to conditional entropy: given what you've seen, how uncertain are you about what strategy to pursue?

If every card works in every deck, there are no signals and the draft is boring. If cards are hyper-specific, signals are clear but drafting is too constrained.

### 4.6 Late-Pick Quality

In many cubes, the last few cards in each pack are unplayable filler. A well-designed cube minimizes dead picks. Measure the average quality of the last 3-5 picks in each pack:

\[
f(C) = \mathbb{E}\left[\text{quality of picks 13-15 in each pack}\right]
\]

---

## Category 5: Information-Theoretic Objectives

### 5.1 Draft Strategy Entropy

Define the set of possible "strategies" a drafter can follow (e.g., color pairs, archetypes). A cube is replayable if many strategies are roughly equally good. This is captured by the entropy of the distribution over optimal strategies across many drafts:

\[
H(\text{strategies}) = -\sum_s P(s\text{ is optimal}) \log P(s\text{ is optimal})
\]

Max entropy = all strategies are equally likely to be the best one in any given draft.

### 5.2 Mutual Information Between Cards

\[
I(i; j) = \log \frac{P(i \text{ and } j \text{ in same deck})}{P(i \text{ in deck}) \cdot P(j \text{ in deck})}
\]

This is essentially pointwise mutual information (PMI). We could use PMI instead of EDHREC's synergy score as the base measure. PMI has a more principled statistical foundation and is symmetric by definition.

### 5.3 Conditional Entropy of Optimal Strategy

Given a drafter's first 3 picks, how uncertain is the optimal completion of their deck?

High conditional entropy = many viable completions (flexible, replayable).
Low conditional entropy = the first few picks determine everything (predictable, boring).

---

## Category 6: Portfolio Theory / Balancing Risk

### 6.1 Markowitz Portfolio

Treat each archetype as an "asset" and each card as an investment that contributes to one or more archetypes. The expected "return" of a card is its synergy contribution; the "risk" is the variance.

\[
f(C) = \mathbb{E}[\text{synergy}(C)] - \lambda \cdot \text{Var}[\text{synergy}(C)]
\]

where variance is over random draft outcomes. This balances "the cube has high synergy on average" against "the cube's synergy is consistent across drafts."

### 6.2 Risk Parity Across Archetypes

Each archetype should contribute equally to the cube's total "risk" (variance in draft outcomes). If one archetype is much higher variance than others (e.g., it's either amazing or unplayable depending on the draft), the cube is unbalanced.

### 6.3 Robust Optimization

Instead of optimizing for the expected case, optimize for the worst case:

\[
f(C) = \min_{\text{draft outcome}} \text{quality}(\text{draft outcome given cube } C)
\]

This ensures no draft is terrible, even if it means the average draft isn't the best possible.

---

## Category 7: Alternative Association Measures

The choice of the *base measure* (what goes into the matrix entries) is separate from the choice of *aggregation* (how we combine pairwise scores). EDHREC "synergy" is one option. There are others.

### 7.1 Pointwise Mutual Information (PMI)

\[
\text{PMI}(i,j) = \log \frac{P(\text{deck has both } i \text{ and } j)}{P(\text{deck has } i) \cdot P(\text{deck has } j)}
\]

PMI is symmetric, grounded in probability theory, and handles rare vs. common cards more naturally than raw percentage differences. Positive PMI = the cards co-occur more than expected. Negative PMI = they co-occur less.

Can be computed from EDHREC's data: `inclusion / potential_decks` gives `P(both | i)`, and base rates can be estimated from overall inclusion rates.

### 7.2 Normalized PMI (NPMI)

\[
\text{NPMI}(i,j) = \frac{\text{PMI}(i,j)}{-\log P(\text{both})}
\]

Ranges from -1 to +1 regardless of base rates. Easier to compare across pairs with very different frequencies.

### 7.3 Lift (already available from EDHREC)

\[
\text{Lift}(i,j) = \frac{P(\text{both})}{P(i) \cdot P(j)}
\]

Lift > 1 means positive association. Already in the data. Could use log(lift) = PMI.

### 7.4 Jaccard Similarity

\[
J(i,j) = \frac{|\text{decks with both}|}{|\text{decks with either}|}
\]

Doesn't separate "synergy" from "both cards are just popular." Two staples that appear in every deck will have high Jaccard similarity regardless of synergy. This might actually be useful for a different purpose: identifying cards that *belong in the same deck* rather than cards that *synergize*.

### 7.5 Cosine Similarity of Deck Vectors

Represent each card as a vector over decks (1 if the card is in the deck, 0 otherwise). The cosine similarity between two card vectors measures how similar their deck inclusion patterns are.

This is related to Jaccard but handles magnitude differently. It would require the raw deck-level data (which we don't have from EDHREC, but could potentially get from other sources like Moxfield or MTGO data).

### 7.6 Confidence-Weighted Synergy

Weight synergy by how confident we are in the measurement:

\[
S'[i,j] = S[i,j] \cdot \left(1 - \frac{1}{\sqrt{\text{num\_decks}[i,j]}}\right)
\]

This down-weights synergy based on small sample sizes and gives more weight to heavily-played cards with reliable data. Avoids the problem where two obscure cards have +80% "synergy" based on appearing in only 3 decks.

---

## Category 8: Hierarchical and Decomposed Objectives

### 8.1 Two-Level: Archetype Composition + Filling

1. **Outer problem:** Choose which archetypes to support and how many cards each gets.
2. **Inner problem:** For each archetype, choose the best cards.

The outer problem is small (maybe 8-12 archetypes, choosing quotas that sum to 360). The inner problem decomposes into independent subproblems, one per archetype. Each inner subproblem is a much smaller version of the current ILP.

Bridge cards (those that contribute to multiple archetypes) would need special handling -- they reduce the effective quota for each archetype they span.

### 8.2 Core + Periphery

First, find the dense core: a set of ~50 "staple" cards that have broad synergy. Then, build outward: add archetype-specific cards that synergize with a subset of the core.

\[
f(C) = \text{synergy}(\text{core}) + \alpha \sum_{\text{arch}} \text{synergy}(\text{arch cards with core})
\]

This mirrors how many human designers build cubes: start with the generically strong cards, then add themed packages.

### 8.3 Layered Construction

Build the cube in explicit layers, each with its own objective:

1. **Foundation (50-60 cards):** Mana fixing, format staples. Optimize for broad inclusion rate.
2. **Archetype anchors (80-100 cards):** High-synergy cards that define archetypes. Optimize for within-archetype synergy.
3. **Role players (100-120 cards):** Cards that support archetypes but aren't archetype-defining. Optimize for coverage (filling gaps in color, curve, type).
4. **Glue (40-60 cards):** Bridge cards that connect archetypes. Optimize for betweenness centrality or multi-archetype synergy.
5. **Spice (20-40 cards):** Unique, interesting cards that create memorable moments. Optimize for... fun? Novelty? Maybe just hand-picked.

### 8.4 Nested Synergy

\[
f(C) = \alpha \sum_{\text{arch}} \text{intra-synergy}(\text{arch}) + \beta \sum_{\text{arch}_1 \neq \text{arch}_2} \text{inter-synergy}(\text{arch}_1, \text{arch}_2)
\]

Reward both tight within-archetype synergy AND some cross-archetype connections. The ratio alpha/beta controls how siloed vs. interconnected the archetypes are. Too siloed = drafts are on rails. Too interconnected = archetypes lose identity.

---

## Category 9: Learning from Existing Cubes

### 9.1 Cube Similarity

CubeCobra hosts thousands of human-designed cubes. We could scrape popular cubes and learn what cards co-occur. Then optimize for similarity to a "composite ideal cube":

\[
f(C) = \text{similarity}(C, \text{average popular cube})
\]

But this just gives us an average cube, not a novel optimal one.

### 9.2 Collaborative Filtering

Treat cubes as "users" and cards as "items." Use matrix factorization (like Netflix recommendations) to learn latent card features. Then select cards that cover the latent factor space.

This bypasses the synergy question entirely -- it learns what cube designers value implicitly from their designs.

### 9.3 Contrastive Learning

Learn what distinguishes "great" cubes (high-rated on CubeCobra) from mediocre ones. Use the learned features as the objective.

---

## Category 10: Constraint-Heavy / Feasibility-First

### 10.1 Pure Constraint Satisfaction

Don't optimize anything. Instead, define enough constraints that any feasible solution is a reasonable cube:

- Color balance: 50-70 cards per color, 30-50 multicolor, 20-40 colorless
- Mana curve: specified distribution across CMCs
- Card types: minimum creatures, minimum instants/sorceries, minimum lands
- Keyword quotas: at least 15 cards with each of 8-10 mechanical themes
- Archetype support: at least 20 cards supporting each of 10 archetypes
- No "dead cards": every card must synergize with at least 5 others in the cube

Then just find *any* feasible solution. If the constraints are good enough, the result is a reasonable cube.

Advantage: fast, no complex objective. Disadvantage: no optimization, so no reason to prefer one feasible solution over another.

### 10.2 Maximum Entropy Feasible

Among all cubes satisfying the constraints, find the one with maximum entropy (most "random" subject to the constraints). This avoids biasing toward any particular strategy while ensuring all constraints are met. Related to the principle of maximum entropy in statistical mechanics.

### 10.3 Constraint Tightening

Start with loose constraints and a synergy objective. Iteratively tighten constraints (add more, make ranges narrower) and re-solve. This lets you see the Pareto frontier between synergy and diversity. At some point, additional constraints start hurting synergy a lot -- that's where the interesting tradeoffs are.

---

## Category 11: Multi-Objective / Pareto Approaches

### 11.1 Weighted Sum

\[
f(C) = w_1 \cdot \text{total\_synergy}(C) + w_2 \cdot \text{color\_balance}(C) + w_3 \cdot \text{mana\_curve\_quality}(C) + \ldots
\]

Simple but requires choosing weights. Different weights give different solutions.

### 11.2 Lexicographic Optimization

1. First, maximize synergy.
2. Among all solutions with near-optimal synergy (within 5%), maximize diversity.
3. Among those, maximize mana curve quality.
4. ...

Establishes a priority ordering over objectives.

### 11.3 Pareto Frontier

Compute the set of non-dominated solutions: cubes where you can't improve one metric without hurting another. Then present the frontier to the human designer to choose from.

This is the most honest approach -- it doesn't pretend there's a single "best" cube, but rather shows the space of tradeoffs.

### 11.4 Goal Programming

Set a target for each metric (e.g., synergy >= 5000, color entropy >= 2.1, archetype count >= 10). Minimize the total shortfall across all targets:

\[
f(C) = -\sum_{g} \max(0, \text{target}_g - \text{actual}_g(C))
\]

This is practical and intuitive: you say what you want, and the optimizer gets as close as it can.

---

## Category 12: Wild Ideas

### 12.1 Narrative Coherence

A cube tells a "story" -- there are themes, callbacks, and interactions that create a sense of world-building. Could we measure this? Maybe by looking at shared creature types, flavor text themes, or plane/set associations. Completely non-standard as an optimization metric, but captures something real about cube design.

### 12.2 Surprise / Discovery Potential

How many non-obvious synergies does the cube contain? Define a synergy as "surprising" if the two cards don't share obvious mechanical keywords but have high empirical synergy. A cube with many surprising synergies is more fun to discover.

\[
f(C) = \sum_{i,j \in C} S[i,j] \cdot \text{surprise}(i,j)
\]

where `surprise(i,j) = 1 - text_similarity(oracle_text_i, oracle_text_j)` or similar.

### 12.3 Metagame Dynamics

In a repeated draft league (same cube, different drafts), does a metagame develop? Do different strategies rise and fall in popularity? This is a sign of a rich, replayable cube. Measuring this requires simulating many drafts with adaptive agents -- computationally heavy but theoretically interesting.

### 12.4 Complexity Budget

Some cards are simple (a 2/2 creature for 2 mana) and some are complex (6 lines of rules text with triggered abilities). A cube needs both. Too many complex cards = cognitive overload. Too many simple cards = boring.

\[
\text{Penalize:} \quad \left|\text{avg\_complexity}(C) - \text{target}\right|
\]

Complexity can be estimated from oracle text length, number of keywords, or rules text features.

### 12.5 Power Level Uniformity

Every card should be approximately equally desirable as a first pick. If some cards are way above the rest, they distort the draft ("just pick the bomb every time"). Measure power level variance and penalize it.

This requires a power level estimate per card. Could use EDHREC inclusion rates as a proxy: cards in more decks are probably more powerful (in Commander at least -- draft power is different).

### 12.6 Combo Density

How many two-card or three-card combos exist in the cube? Combos make drafts exciting (assembling the combo feels great). Too many combos = chaotic. Too few = boring.

Could define a target combo density and optimize toward it.

---

## Choosing an Approach: Axes to Consider

| Axis | Range |
|---|---|
| **Interpretability** | Weighted sum (easy to explain) ← → DPP (need a PhD) |
| **Computational cost** | Constraint satisfaction (fast) ← → Draft simulation (very slow) |
| **Data requirements** | Current synergy matrix only ← → Full draft simulation engine + AI drafters |
| **Human input** | Zero (fully automated) ← → Heavy (human defines archetypes, weights, targets) |
| **Theoretical grounding** | Ad hoc (threshold synergy) ← → Deep (information theory, portfolio theory) |
| **Flexibility** | Single objective ← → Pareto frontier with human selection |
| **Scale** | Works at k=10 ← → Works at k=360 |

---

## Promising Combinations

Not committing to anything, but some combinations seem worth exploring further:

**Pragmatic baseline:** Total synergy (1.1) + hard constraints (color, curve, type) + keyword quotas. This is the simplest upgrade from the current approach. Add constraints to prevent monoculture, keep synergy as the objective.

**Balanced approach:** Log-synergy (1.2) + archetype coverage (community detection) + min color pair synergy (3.4). The log prevents monoculture naturally, community detection gives us archetypes, and the color pair constraint ensures all draft lanes are viable.

**Draft-centric:** Build a draft simulator first, then use draft branching factor (4.2) + drafter equity (4.3) + average deck quality (4.1) as a composite objective. Most realistic but highest implementation cost.

**Decomposed:** Two-level optimization (8.1) using spectral clustering for archetype discovery, then greedy filling with submodular coverage (3.1). Scales well and gives interpretable results.

**Data-driven:** Collaborative filtering (9.2) from CubeCobra data to learn card embeddings, then facility location (3.3) in the embedding space. Bypasses the synergy question but requires a new data source.

---

## Open Questions

1. **How important is synergy vs. raw power level?** A cube of all mediocre but synergistic cards might be worse than a cube of powerful cards with moderate synergy. Should we incorporate power level at all?

2. **How much should we trust EDHREC synergy data for cube design?** Commander is a different format (100-card singleton, multiplayer). Synergy patterns in Commander may not transfer to cube drafting (40-card decks, 1v1). Is EDHREC data even the right signal?

3. **Should archetypes be discovered or specified?** Spectral clustering finds "natural" archetypes in the data. But a cube designer might want specific archetypes (e.g., "I want an aristocrats archetype"). Top-down vs. bottom-up.

4. **Do we need a draft simulator?** The simulation-based objectives are the most realistic but require building a whole separate system. Is it worth it, or can we get 80% of the benefit from static metrics?

5. **What's the right scale for the pre-filter?** Even if we pick the perfect objective, we need to solve it over thousands of cards. The pre-filter determines the candidate pool. Should it be synergy-based, diversity-based, or something else?

6. **How do we evaluate our objective function?** Whatever we choose, how do we know it produces good cubes? Eventually we need human evaluation. What's the minimum viable "is this cube any good?" test?
