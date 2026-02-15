# Data Logistics: Bulk Downloading EDHREC

## Current State of Our Data

**We have a problem.** The `downloaded_cards.txt` file claims 26,672 cards were downloaded, but there are only **39 files** in `cards/`. The rest were lost -- likely deleted or never committed (the `cards/` directory is gitignored). The `seen_cards.txt` file (53,845 entries with frequency counts) is intact and checked into git, so we know which cards exist in the EDHREC ecosystem. But the actual synergy data files are essentially gone.

The 39 surviving card files are from **January 9, 2026** and contain ~340 lines each on average (716KB total). They're already somewhat stale.

There are also 5 files with **0 bytes** (cards with apostrophes like `kodama-s-reach` -- a name formatting bug that generated wrong URLs).

**Bottom line: we're starting from scratch on the synergy data.**

---

## The JSON API (Much Better Than HTML Scraping)

The current scraper (`download_card.py`) fetches HTML pages from `edhrec.com/cards/{name}`, converts them to markdown, and regex-parses the results. This is fragile and lossy.

EDHREC has an **undocumented JSON API** that returns clean, structured data:

```
https://json.edhrec.com/pages/cards/{card-name}.json
```

The JSON response includes categorized card lists with structured fields:

| Field | Example | Meaning |
|---|---|---|
| `name` | `"Brainstorm"` | Card name |
| `sanitized` | `"brainstorm"` | URL slug |
| `synergy` | `0.42` | Synergy score (as decimal, not percentage) |
| `lift` | `2.93` | Statistical lift (how much more likely than random) |
| `inclusion` | `99187` | Number of decks containing both cards |
| `num_decks` | `99187` | Same as inclusion |
| `potential_decks` | `188132` | Total decks in the reference population |

The card lists are categorized into sections like:
- **Top Cards** -- most commonly played alongside this card
- **High Synergy Cards** -- highest synergy scores
- **High Lift Cards** -- highest statistical lift
- **New Cards** -- recently released cards
- **Top Commanders** -- commanders this card appears with

**Advantages over HTML scraping:**
- No HTML-to-markdown conversion, no fragile regex parsing
- Get `lift` and `potential_decks` fields that the HTML scraper misses
- Structured categories (synergy vs co-occurrence vs lift)
- Card metadata included (Scryfall ID, color identity, CMC, type, rarity)
- Price data included (though we don't need it)

---

## Rate Limiting & Politeness

### What We Know

- `robots.txt` does **not** disallow `/cards/` or the JSON API. The only disallowed paths are `/articles/preview/`, `/articles/search/`, and `/deckpreview/`.
- EDHREC's Terms of Service grant a "non-transferable, non-exclusive, revocable, limited license to access the Site solely for your own personal, noncommercial use."
- The ToS also says: "you shall not access the Site in order to build a similar or competitive website."
- There is no documented rate limit.

### Recommendation

This is a personal, noncommercial project -- not a competing website. But we should still be respectful:

- **1 request per second** is a reasonable baseline. Not so aggressive that we'd stress their servers, fast enough to be practical.
- **0.5 requests per second** (1 every 2 seconds) is more conservative and probably the safest choice.
- Include a proper `User-Agent` header identifying the project.
- Run during off-peak hours if possible.

### Time Estimates

| Cards to Download | Rate | Wall Clock Time |
|---|---|---|
| 5,000 (top cards only) | 1/sec | ~1.4 hours |
| 5,000 (top cards only) | 0.5/sec | ~2.8 hours |
| 26,000 (previous coverage) | 1/sec | ~7.2 hours |
| 26,000 (previous coverage) | 0.5/sec | ~14.4 hours |
| 53,800 (all known cards) | 1/sec | ~15 hours |
| 53,800 (all known cards) | 0.5/sec | ~30 hours |

### Do We Need All 53,800?

**Probably not.** The long tail of `seen_cards.txt` includes extremely obscure cards with minimal synergy data. The top 5,000-10,000 cards by frequency would cover the vast majority of the synergy graph's useful edges. Cards seen only a handful of times won't contribute meaningful synergy data anyway.

A practical strategy:
1. Download the top ~5,000 cards from `seen_cards.txt` (the ones with highest frequency counts). These are the cards that appear most often in other cards' synergy lists, so they have the richest data. **~1.5-3 hours.**
2. Build matrices and run analysis on that set.
3. If coverage feels thin, expand to 10,000-15,000. 

---

## Storage & Format

### JSON Files

Each JSON response is roughly **20-100KB** depending on how many related cards are listed. For 5,000 cards:
- **~250-500MB** of raw JSON files
- This is very manageable

### Recommendation: Save Raw JSON

Save the raw JSON responses to `cards_json/{card-name}.json` rather than the parsed text format. This preserves all fields for future use and avoids re-downloading if we want to extract different data later. The matrix-building script can parse JSON directly.

---

## Scryfall: The Other Data Source We Need

EDHREC gives us **synergy relationships** (which cards go well together). But for constraint-based optimization (color balance, mana curve, keyword quotas), we need **card metadata**: color identity, mana cost, oracle text, keywords, type line, etc.

**Scryfall** provides exactly this via bulk data downloads:

```
https://api.scryfall.com/bulk-data/oracle-cards
```

This returns a single ~161MB JSON file with every unique card, including:
- `color_identity`: `["R"]`
- `cmc`: `1.0`
- `mana_cost`: `"{R}"`
- `type_line`: `"Instant"`
- `oracle_text`: `"Lightning Bolt deals 3 damage to any target."`
- `keywords`: `[]`
- `rarity`: `"common"`

**This is a one-time download** that we can do immediately. No rate limiting concerns -- Scryfall explicitly provides this for bulk use.

With both data sources combined:
- **EDHREC** → synergy matrix (which cards work together)
- **Scryfall** → card properties (for constraints: color, CMC, type, keywords)

---

## Revised Scraper Design

The current `scrape_edhrec.py` / `download_card.py` needs a rewrite to:

1. **Use the JSON API** instead of HTML scraping
2. **Save raw JSON** instead of parsed text
3. **Download by priority** from `seen_cards.txt` (top N by frequency)
4. **Track progress** with proper resume capability (check which JSON files already exist on disk)
5. **Rate limit** with configurable delay (default 1-2 seconds between requests)
6. **Handle the apostrophe bug** -- the current `format_card_name` strips apostrophes, producing `kodama-s-reach` instead of `kodamas-reach`. Check which format EDHREC actually expects.

---

## Legal Summary

| Factor | Assessment |
|---|---|
| robots.txt | `/cards/` is **allowed** |
| ToS: personal noncommercial use | **We qualify** |
| ToS: not building a competing website | **We qualify** (optimization tool, not a card database) |
| Rate limiting | **Be polite** -- 1 req/sec or slower |
| Data redistribution | **Don't redistribute** raw EDHREC data |

The main risk is that the JSON API is undocumented, so EDHREC could block it or change it at any time. Saving raw JSON locally mitigates this -- we only need to download once.

---

## Implementation Plan

### Step 1: Download Scryfall Bulk Data (immediate, ~1 minute)

Download the Oracle Cards bulk file from Scryfall. This is a single HTTP request that gives us every card's metadata (color identity, CMC, type, keywords, oracle text). Save it to `data/scryfall_oracle_cards.json`.

```
GET https://api.scryfall.com/bulk-data/oracle-cards
```

This returns a JSON object with a `download_uri` field pointing to the actual data file. Fetch that URI to get the full dataset.

Build a lookup table keyed by card name (normalized) so we can join EDHREC synergy data with Scryfall metadata later.

### Step 2: Rewrite the Scraper (~1-2 hours of dev work)

Replace `download_card.py` and update `scrape_edhrec.py`:

**New `download_card.py`:**
- Fetch from `https://json.edhrec.com/pages/cards/{card-name}.json`
- Return the raw JSON (parsed as a Python dict)
- Retry with exponential backoff (keep existing logic)
- Add a proper `User-Agent` header

**New `scrape_edhrec.py`:**
- Read the top N cards from `seen_cards.txt` (sorted by frequency, which it already is)
- For each card, check if `cards_json/{card-name}.json` already exists on disk (resume capability)
- If not, download via the new `download_card.py` and save the raw JSON
- Sleep 1-2 seconds between requests
- Print progress (e.g., `"Downloaded 142/5000: lightning-bolt"`)
- Handle Ctrl+C gracefully (the data saved so far is safe because we save after each card)

**Directory layout:**
```
data/
  scryfall_oracle_cards.json   # Scryfall bulk data (one-time download)
cards_json/                     # EDHREC JSON responses (one per card)
  lightning-bolt.json
  sol-ring.json
  ...
```

### Step 3: Run the Download (~1.5-3 hours wall time)

Start with the top 5,000 cards from `seen_cards.txt`. At 1 request/second, this takes ~1.4 hours. At 2 seconds between requests, ~2.8 hours. This can run in the background.

**Verification:** After downloading, spot-check a few JSON files to confirm they contain the expected synergy data. Count the total files and compare against the target.

### Step 4: Rewrite Matrix Construction

Update `load_cards_files.py` to:
- Read from `cards_json/*.json` instead of `cards/*.txt`
- Parse the structured JSON (no more regex)
- Extract `synergy`, `lift`, `inclusion`, `potential_decks` per card pair
- Build matrices as before, but also build a **lift matrix** and a **confidence matrix** (based on `potential_decks`)
- Symmetrize the synergy matrix: `S' = (S + S^T) / 2` for entries where both directions exist; keep the single direction where only one exists
- Save the enriched matrices

### Step 5: Build the Card Metadata Table

Join EDHREC card names with Scryfall data to produce a per-card properties table:

| Field | Source |
|---|---|
| `name` | EDHREC / Scryfall |
| `color_identity` | Scryfall |
| `cmc` | Scryfall |
| `type_line` | Scryfall |
| `keywords` | Scryfall |
| `oracle_text` | Scryfall |
| `rarity` | Scryfall |
| `edhrec_frequency` | `seen_cards.txt` |

This table is what the optimizer will use for constraint-based filtering (color balance, mana curve, keyword quotas).

### Decision Point: How Many Cards?

After Steps 1-5, we'll have real data and can evaluate:
- How dense is the synergy graph at 5,000 cards?
- Are there obvious cluster structures (archetypes)?
- Do we need more coverage?

If the graph is too sparse, expand to 10,000-15,000 cards by running the scraper again with a higher N.
