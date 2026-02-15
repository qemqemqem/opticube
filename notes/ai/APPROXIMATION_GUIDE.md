# Approximation Guide: Working with Opticube's Solvers

How to control effort, interpret results, and know when to trust the output.

---

## The Two Worlds

Opticube has two fundamentally different kinds of solvers. Understanding the difference is the single most important thing for interpreting results.

### Exact solvers (ILP): CBC, HiGHS, CP-SAT

These solve the problem *mathematically*. They explore a search tree, maintain a **dual bound** (theoretical ceiling on the best possible solution), and can *prove* how close their answer is to optimal.

When an exact solver says "optimal" — it means it. The solution is the best one that exists, or provably within a stated tolerance.

**The catch:** they only work for the *linear* objective (total pairwise synergy). They cannot express `log(1+x)`, ReLU, or soft quadratic penalties. And they blow up in runtime when the problem gets large (~500+ cards with O(n²) auxiliary variables).

### Heuristic solvers: Greedy + Local Search

These use smart guessing. Greedy construction picks one card at a time by marginal gain. Local search tries random swaps and keeps improvements. Multiple restarts explore different starting points.

Heuristic solvers have **no gap, no bound, no proof**. The solution could be optimal, or it could be 20% below optimal — there's no way to know from the solver output alone.

**The upside:** they handle *any* objective function (log-synergy, penalties, whatever) and scale to thousands of cards.

---

## Effort Parameters

### For exact solvers (ILP)

| Parameter | CLI Flag | What it controls |
|---|---|---|
| **Time limit** | `--time-limit 120` | Max wall-clock seconds. Solver stops and returns best solution found. |
| **Gap tolerance** | `--gap-rel 0.05` | Stop when proven within X% of optimal. Lower = more time for better guarantee. |
| **Solver backend** | `--solver cpsat` | Which engine. CP-SAT > HiGHS > CBC for our problem. |
| **Threads** | `--threads 4` | Parallel workers (CP-SAT, HiGHS). CBC is single-threaded. |

**The key tradeoff:** `--gap-rel 0` means "prove exact optimum" — this can take much longer but gives you a perfect answer. `--gap-rel 0.05` means "stop as soon as you've proven within 5%" — much faster, barely worse.

**Time limit is a hard stop.** If the solver hasn't proven optimality within the time limit, you get the best solution found so far. The status will say "feasible" (not "optimal"), meaning: "this is a valid solution but we couldn't prove how good it is."

### For heuristic solvers (Greedy + Local Search)

| Parameter | CLI Flag | What it controls |
|---|---|---|
| **Restarts** | `--num-restarts 10` | Independent runs from different starting points. More = better coverage of solution space. |
| **Patience** | `--max-no-improve 2000` | Swaps to try without improvement before giving up on a restart. More = deeper local search. |
| **Seed** | `--seed 42` | Random seed. Same seed = same result. Change seed to explore different solutions. |

**There is no gap tolerance for heuristics.** They run until patience expires, not until a quality threshold is met.

**Rough guide for `--max-no-improve`:**
- 500: quick and dirty, for iteration
- 2000: reasonable default
- 5000-10000: serious optimization run
- 50000: overnight run, diminishing returns likely

**Rough guide for `--num-restarts`:**
- 3: testing
- 5-10: normal use
- 20-50: when quality matters and you have time

### Is "time spent" a generic parameter?

**For ILP: yes.** `--time-limit` directly controls wall time. The solver uses every second productively (pruning the search tree, tightening the bound).

**For greedy: not directly.** Time is controlled indirectly through `--num-restarts` and `--max-no-improve`. The actual wall time depends on problem size, matrix density, and how quickly restarts converge. You can estimate: each restart with `max_no_improve=2000` on a 500-card pool takes roughly 1-5 seconds. So `--num-restarts 10 --max-no-improve 2000` ≈ 10-50 seconds.

A future improvement (simulated annealing) would make the heuristic more time-controllable through temperature schedules.

---

## What the Solvers Report

### Solver Report (printed after every run)

```
### Solver Report
  Method: CPSAT
  Status: optimal
  Objective: 45.3200
  Wall time: 0.34s
  Optimality gap: ~0% (proven optimal)
  Best bound: 45.3200
  Selected 10 cards
```

### Field-by-field interpretation

**Status:**
- `optimal` — Solution is within `gap_rel` of the true optimum. For CP-SAT with status=OPTIMAL, gap is exactly 0.
- `feasible` — A valid solution was found but optimality wasn't proven (usually hit time limit).
- `heuristic` — Greedy solver. No optimality claim.
- `infeasible` — No valid solution exists (e.g., constraints are contradictory).

**Optimality gap:**
- `~0% (proven optimal)` — This IS the best solution. Trust it completely.
- `≤ 5.00%` — The true optimum is at most 5% better than this. Very reliable.
- `N/A (heuristic)` — No gap information. Could be optimal, could be far from it.

**Best bound:**
- The theoretical ceiling on the objective. No solution can score higher than this.
- When `best_bound ≈ objective`, you're at or near optimum.
- Only available from exact solvers (CP-SAT reports this; PuLP/CBC doesn't expose it).

**Wall time:** How long the solver actually ran. Useful for budgeting time on larger runs.

**Iterations:** For greedy, total swap attempts across all restarts. Higher = more thorough search.

---

## Solver Selection Guide

### CP-SAT (recommended for exact solving)

```bash
python optimizer.py -k 10 --method ilp --solver cpsat
```

- Strongest open-source solver for binary combinatorial problems
- Reports *exact* gap (not just "within tolerance")
- Multi-threaded by default
- Requires integer coefficients internally (floats are scaled by 10000 — precision loss is negligible)

**Use when:** pool ≤ 500 cards, linear objective, you want a provably optimal answer.

### HiGHS

```bash
python optimizer.py -k 10 --method ilp --solver highs --gap-rel 0.01
```

- State-of-the-art MIP solver, strong on general problems
- Multi-threaded via `--threads`

**Use when:** CP-SAT is unavailable, or you want a second opinion on the same problem.

### CBC

```bash
python optimizer.py -k 10 --method ilp --solver cbc
```

- Bundled with PuLP — zero installation needed
- Single-threaded, slower than CP-SAT
- Adequate for small problems (n < 200)

**Use when:** You don't have `ortools` or `highspy` installed.

### Greedy + Local Search

```bash
python optimizer.py -k 360 --method log-synergy --num-restarts 10 --max-no-improve 5000
```

- Only option for the log-synergy objective
- Only option that scales to large pools (1000+ cards)
- No quality guarantee

**Use when:** log-synergy objective, large pool, or ILP is too slow.

---

## How to Build Confidence in Heuristic Results

Since greedy has no gap, here are practical ways to assess quality:

### 1. Compare against ILP on small problems

Run both on the same small problem (k=10, n=100). If greedy finds a solution within 1-2% of the ILP optimum, the heuristic is working well.

```bash
# Exact answer
python optimizer.py -k 10 --method ilp --solver cpsat --objective linear
# Heuristic answer
python optimizer.py -k 10 --method greedy --num-restarts 20 --objective linear
```

Compare the objective values. The gap between them tells you how much the heuristic is leaving on the table.

### 2. Multiple seeds

Run the same problem with different seeds and compare scores:

```bash
python optimizer.py -k 30 --seed 1 --num-restarts 5
python optimizer.py -k 30 --seed 2 --num-restarts 5
python optimizer.py -k 30 --seed 3 --num-restarts 5
```

If all runs give similar scores (within 1-2%), you're likely near a good solution. High variance across seeds = the search space has many local optima and more restarts are needed.

### 3. Diminishing returns curve

Increase `--num-restarts` and watch the score:

```
restarts=3:  score=142.5
restarts=10: score=145.1
restarts=30: score=145.8
restarts=50: score=146.0
```

When doubling restarts barely improves the score, you're in the flat part of the curve. Good enough.

### 4. Score decomposition

The solver reports score breakdown (synergy term, penalties). Look at these — a good cube has high synergy *and* reasonable penalties. If synergy is great but color balance is terrible, tune lambdas rather than spending more compute.

---

## Common Pitfalls

**"Optimal" doesn't mean "good."** ILP finds the mathematically best solution for the given objective. If the objective doesn't capture what you care about (e.g., missing penalty terms), the "optimal" solution might be terrible as a cube.

**Gap tolerance ≠ solution quality.** A 5% gap means the *objective value* is within 5% of optimal. It doesn't mean the *card selection* is 95% identical to the true optimum. Two solutions with nearly identical scores can have very different card compositions.

**Greedy is not random.** The greedy construction phase is deterministic given a seed. It finds surprisingly good solutions even without local search. The local search phase then polishes.

**Scaling is the real constraint.** A 500-card pool with ILP creates ~125,000 auxiliary binary variables. This is tractable but slow. At 2000 cards, it's ~2 million variables — CBC will struggle, CP-SAT might handle it, greedy won't blink.

---

## Summary Table

| Question | ILP (CBC/HiGHS/CP-SAT) | Greedy + Local Search |
|---|---|---|
| Can I trust the result? | Yes — proven within gap tolerance | Maybe — compare across runs |
| How close to optimal? | Gap tells you exactly | Unknown |
| Can I control time? | Yes: `--time-limit` | Indirectly: `--num-restarts`, `--max-no-improve` |
| What objective? | Linear pairwise synergy only | Anything (log, penalties, etc.) |
| Max problem size? | ~500 cards practical | 5000+ cards |
| Key knob for quality | `--gap-rel` (lower = better proof) | `--num-restarts` (more = better coverage) |
| Key knob for speed | `--time-limit` (higher = more time) | `--max-no-improve` (lower = faster per restart) |
