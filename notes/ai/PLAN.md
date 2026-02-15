# Opticube — Project Plan

---

## 0. Problem Statement

Given a **pool** of N candidate cards (currently N=500), select a **cube** of K cards that maximizes incoming log-synergy:

$$f(C) = \sum_{A \in C} \ln\!\left(1 + \sum_{B \in C,\, B \neq A} \operatorname{ReLU}(S[B, A])\right)$$

subject to soft quadratic penalties for color balance, type balance, and mana curve.

- **S[B, A]** = synergy of card A on card B's EDHREC page ("how much does B want A around?")
- **ReLU** = ignore anti-synergy (clip negatives to zero)
- **ln(1 + x)** = diminishing returns — prevents monoculture, spreads synergy across many cards

Starting with **K=10** for validation (small enough to inspect every card by hand), scaling to K=30, then K=180, then K=360.

Full spec with penalty terms, targets, and lambdas: [OBJECTIVE_V1.md](./OBJECTIVE_V1.md)

Design rationale: [OBJECTIVE_FUNCTION_REFINED.md](./OBJECTIVE_FUNCTION_REFINED.md)

---

## 1. Status

| Component | Status | Notes |
|---|---|---|
| Scryfall metadata | **Done** | `data/scryfall_oracle_cards.json` — 36,709 cards |
| Card metadata table | **Done** | `data/card_metadata.json` — Scryfall + EDHREC merged (98.6% match rate) |
| EDHREC synergy data | **Done (500 cards)** | `cards_json/{card-name}.json`, top 500 by frequency |
| Matrix construction | **Done** | Sparse N x N matrices, pool-scoped. 222k non-zero entries at N=500 |
| Incoming synergy matrix | **Done** | ReLU'd, ready for log-synergy objective via `synergy.load_incoming_synergy()` |
| ILP/CP-SAT solvers | **Done** | Linear objective only (baseline/sanity check) |
| Greedy + local search | **Done** | But uses old linear objective, not log-synergy |
| **Log-synergy objective** | **NOT IMPLEMENTED** | Spec is ready, data is ready, solver needs updating |
| **Simulated annealing** | **NOT IMPLEMENTED** | Local search is pure hill-climbing, gets stuck in local optima |

The two **not implemented** items are the next tasks — in that order.

---

## 2. Next Steps

### Step 1: Implement the log-synergy objective

Wire the log-synergy formula into the greedy + local search solver. Requires:

1. `evaluate(cube)` — compute full f(C) with ln, ReLU, and penalty terms
2. `evaluate_swap(cube, card_out, card_in)` — compute delta efficiently
3. Card property lookup for color/type/curve penalties (data ready in `card_metadata.json`)
4. Default lambda values and target distributions (specified in [OBJECTIVE_V1.md](./OBJECTIVE_V1.md))

ILP/CP-SAT cannot express the log objective. They stay as baselines on the linear objective only.

### Step 2: Proof-of-concept run (K=10, N=500)

Run with the ~500 cards we already have. K=10 is small enough to inspect every card by hand.

- Does the optimizer produce output? Does it terminate?
- Do the card names look reasonable together?
- Start with all lambdas = 0 (pure synergy, no balance penalties)

### Step 3: Inspect and sanity-check

Look at the K=10 output:

- Is it all one color? One archetype? One CMC?
- Do the cards make sense together, or is it gibberish?
- What's the synergy score? What's the color distribution?

### Step 4: Turn on balance penalties

Set lambdas to nonzero values and re-run. Compare output to the pure-synergy run.

- Does color distribution improve?
- Does type balance improve?
- How much synergy do we give up?

Iterate on lambda values until the output looks like a sensible small selection.

### Step 5: Add simulated annealing

Convert hill-climbing to simulated annealing (accept worsening swaps with probability `exp(-delta/T)`, cool T over time). This is the single highest-impact change for solution quality. No new library needed — just a change to `find_good_subset_greedy` in `optimizer.py`.

### Step 6: Scale up

- Increase K toward 30, then 180, then 360
- Increase pool to 2,000–5,000 cards (requires more EDHREC downloads)
- Monitor solver runtime and quality
- Tune lambdas at each scale

---

## 3. Data Pipeline

Three data sources, all working. Run the pipeline with:

```bash
python run_pipeline.py --top 10              # tiny test (~10s)
python run_pipeline.py --top 50 --delay 0.5  # small test (~30s)
python run_pipeline.py --top 500             # medium (~10 min)
python run_pipeline.py --top 5000            # full (~1.5 hours)
python run_pipeline.py --skip-download       # rebuild matrices from local data
```

Caching: Scryfall re-downloads after 7 days (`--max-age`). EDHREC card files re-download after 14 days. Use `--force` to bypass.

### Matrices produced

| File | Contents |
|---|---|
| `incoming_synergy_matrix.pkl` | **Primary.** M[b,a] = max(0, S[b,a]). Column sums = total incoming synergy. |
| `synergy_matrix.pkl` | Raw asymmetric S[B,A] (may be negative) |
| `synergy_symmetric_matrix.pkl` | Averaged where both directions exist |
| `lift_matrix.pkl`, `inclusion_matrix.pkl`, `potential_decks_matrix.pkl` | Alternative measures |
| `card_names.pkl` / `card_name_to_id.pkl` | Index mapping |

Pool scoping: matrices are N x N for the top N cards, controlled by `pool_size` in `build_matrices()`.

Background: [DATA_LOGISTICS.md](./DATA_LOGISTICS.md)

---

## 4. Accessing the Data

### Load incoming synergy matrix (recommended)

```python
from synergy import load_incoming_synergy, incoming_synergy_for

matrix, card_names, name_to_id = load_incoming_synergy()

# Total incoming synergy for a card (sum of its column)
total = incoming_synergy_for("sol-ring", matrix, card_names, name_to_id)

# Specific pair
i, j = name_to_id['krenko-mob-boss'], name_to_id['goblin-chieftain']
print(matrix[i, j])  # ReLU'd synergy of chieftain on krenko's page
```

### Card name slugs

All card names are normalized slugs: `format_card_name("Kodama's Reach")` -> `kodamas-reach`. Use `utils.format_card_name()` to convert display names.

---

## 5. Optimizer

`optimizer.py` — multiple solving backends.

| Problem | Solver | Why |
|---|---|---|
| Small pool, linear objective | **CP-SAT** | Exact optimum, sub-second |
| Small pool, quick sanity check | **CBC** | Zero setup, good enough |
| Any size, any objective | **Greedy + local search** | Only thing that scales and handles log |

**Current objective:** Total pairwise synergy (old linear). Needs updating to log-synergy (step 1 above).

Spectral tools (farthest-point sampling, clustering) are in `optimizer.py` as preprocessing/analysis tools, not solvers.

### Dependencies

```
pip install pulp        # CBC (bundled)
pip install highspy     # HiGHS API
pip install ortools     # CP-SAT
```

Tests: `tests/test_optimizer.py`

How to control effort, interpret gaps, and trust results: [APPROXIMATION_GUIDE.md](./APPROXIMATION_GUIDE.md)

Earlier redesign notes: [OPTIMIZER_NEXT_STEPS.md](./OPTIMIZER_NEXT_STEPS.md)

---

## 6. Reference Docs

| Doc | Contents |
|---|---|
| [OBJECTIVE_V1.md](./OBJECTIVE_V1.md) | Full objective spec: formula, penalty terms, targets, lambdas, solver algorithm |
| [OBJECTIVE_FUNCTION_REFINED.md](./OBJECTIVE_FUNCTION_REFINED.md) | Design rationale, alternatives considered, future extensions |
| [OBJECTIVE_FUNCTION_BRAINSTORM.md](./OBJECTIVE_FUNCTION_BRAINSTORM.md) | Wide brainstorm of all options explored |
| [DATA_LOGISTICS.md](./DATA_LOGISTICS.md) | Data pipeline background and logistics |
| [OPTIMIZER_NEXT_STEPS.md](./OPTIMIZER_NEXT_STEPS.md) | Optimizer bug inventory and redesign notes |
| [MATHEMATICS.md](./MATHEMATICS.md) | Mathematical background |
| [APPROXIMATION_GUIDE.md](./APPROXIMATION_GUIDE.md) | How to interpret solver output, control effort, trust results |
