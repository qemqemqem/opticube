# Opticube - Project Overview

**Goal:** Automatically build an optimal MTG (Magic: The Gathering) cube by selecting a subset of cards that maximizes total pairwise synergy, using data scraped from EDHREC.

---

## What Is a Cube?

In MTG, a "cube" is a curated collection of cards used for drafting. The quality of a cube depends heavily on how well cards work *together* -- their synergy. Opticube tries to automate this curation by framing it as a mathematical optimization problem.

---

## Pipeline

The project has three phases that run as separate scripts:

### Phase 1: Data Collection (`scrape_edhrec.py` + `download_card.py`)

A web crawler that scrapes card relationship data from [EDHREC](https://edhrec.com), a site that aggregates data from real Commander decks.

**How it works:**

1. Starts with a seed card (default: Lightning Bolt).
2. Downloads the card's EDHREC page, which lists related cards with synergy data.
3. Parses each related card entry. Lines look like:
   ```
   Solphim, Mayhem Dominus: 16% of 131024 decks +14% synergy
   ```
   This means: of the ~131k decks containing the outer card, 16% also contain Solphim, and Solphim has +14% synergy (appears 14% *more* often than expected by chance).
4. Each newly discovered card gets added to a **priority queue** ordered by how frequently the card has been seen across all downloaded pages. High-frequency cards get downloaded first -- the idea being popular cards are more likely to have rich synergy data.
5. Saves each card's relationship data to `cards/{card-name}.txt`.
6. Tracks state across runs in `seen_cards.txt` (card name + frequency count) and `downloaded_cards.txt`.

**Scale:** The `seen_cards.txt` file has ~53,800 entries. `downloaded_cards.txt` has ~26,600 entries. Only a handful of card files are checked into git (the rest are gitignored).

**Utility scripts:**
- `utils.py` -- normalizes card names to URL-friendly slugs (lowercase, ASCII, hyphens).
- `fix_seen_cards.py` -- deduplicates and sorts `seen_cards.txt`.
- `download_card.py` -- downloads a single card page with retry/backoff, parses HTML to markdown, extracts card relationship lines.

### Phase 2: Matrix Construction (`load_cards_files.py`)

Converts the raw text files into three sparse matrices:

| Matrix | Entry `[i, j]` means... |
|---|---|
| **Synergy Matrix** | The synergy percentage between card `i` and card `j` (e.g., +14% means card `j` appears 14% more often than expected in decks containing card `i`) |
| **Percentage Matrix** | The raw co-occurrence rate -- what % of decks containing card `i` also contain card `j` |
| **Num Decks Matrix** | How many total decks were in the sample for card `i`'s page when card `j` appeared |

All three are built as `scipy.sparse.lil_matrix` (for efficient incremental construction), then converted to `csr_matrix` (for efficient arithmetic), and pickled to disk.

**Important detail:** These matrices are **asymmetric**. The entry `synergy[i, j]` comes from card `i`'s EDHREC page and represents `j`'s synergy relative to `i`. The entry `synergy[j, i]` comes from card `j`'s page and may differ (or not exist at all). The analytics module checks this asymmetry empirically.

### Phase 3: Optimization (`analytics.py`)

Loads the matrices and solves for the best subset of cards.

**Three approaches were attempted:**

1. **Random sampling** (`find_good_subset`) -- Randomly picks `set_size` cards, scores the subset by summing all pairwise synergy values, repeats many times, keeps the best. Simple but slow and unlikely to find the optimum.

2. **Quadratic programming** (`find_good_subset_optimized_quadratic`) -- Tried using `cvxpy` to solve as a quadratic program. **Did not work** -- the solver errors out because it can't handle mixed-integer quadratic problems with the available solvers.

3. **Integer Linear Programming** (`find_good_subset_ilp`) -- **The working approach.** Linearizes the quadratic objective using auxiliary binary variables and solves with PuLP's CBC solver. This is the method used in `main()`.

**Current configuration:**
- `set_size = 10` (select 10 cards)
- Before solving, the matrix is **reduced** to the top `set_size * 10 = 100` cards by total synergy (row sum + column sum). This is a heuristic to make the ILP tractable.
- Solver gets a 10-second time limit and a 5% relative gap tolerance for early stopping.

---

## File Inventory

| File | Purpose |
|---|---|
| `scrape_edhrec.py` | Main crawler -- orchestrates downloading and priority queue |
| `download_card.py` | Downloads + parses a single EDHREC card page |
| `load_cards_files.py` | Builds sparse matrices from downloaded card files |
| `analytics.py` | Matrix analysis + optimization (ILP solver) |
| `utils.py` | Card name normalization |
| `fix_seen_cards.py` | Deduplicates/sorts `seen_cards.txt` |
| `seen_cards.txt` | Card name + frequency count (how often each card appeared in other cards' lists) |
| `downloaded_cards.txt` | Set of cards whose EDHREC pages have been downloaded |
| `failed_downloads.txt` | Cards that failed to download |
| `cards/` | One `.txt` file per downloaded card with relationship data (gitignored) |
| `*.pkl` | Pickled matrices and mappings (gitignored) |

---

## Current State / Limitations

- The project is functional but was clearly an experiment in progress.
- Only 10 cards are selected -- the `set_size` is hardcoded.
- The matrix reduction heuristic (top 100 cards) is aggressive and may exclude good candidates.
- The synergy matrix is asymmetric, but the ILP only considers pairs `(i, j)` where `i <= j`, so it effectively only uses one direction of synergy per pair.
- No color identity filtering, mana curve considerations, or other cube design constraints are applied -- it's purely maximizing raw synergy.
- The quadratic programming approach is broken and commented out.
- There's no `requirements.txt` or dependency management file.
