"""
Step 5: Build the card metadata table.

Joins EDHREC card names (from seen_cards.txt) with Scryfall bulk data
to produce a per-card properties table for constraint-based optimization.

Output fields per card:
- name: display name
- sanitized: URL slug
- color_identity: list of colors (e.g., ["R", "G"])
- cmc: converted mana cost
- mana_cost: mana cost string (e.g., "{2}{R}{G}")
- type_line: full type line (e.g., "Creature — Elf Druid")
- keywords: list of keywords (e.g., ["Flying", "Trample"])
- oracle_text: rules text
- rarity: rarity string
- edhrec_frequency: frequency count from seen_cards.txt

Saves to data/card_metadata.json and card_metadata.pkl.
"""

import json
import pickle
from pathlib import Path

from utils import format_card_name

SCRYFALL_PATH = Path("data/scryfall_oracle_cards.json")
SEEN_CARDS_FILE = Path("seen_cards.txt")
OUTPUT_JSON = Path("data/card_metadata.json")
OUTPUT_PKL = Path("card_metadata.pkl")


def build_scryfall_lookup() -> dict[str, dict]:
    """
    Build a lookup table from Scryfall data, keyed by normalized card name.

    For double-faced / split cards, we index by both the full name and the
    front face name, since EDHREC typically uses just the front face.
    """
    print(f"Loading Scryfall data from {SCRYFALL_PATH}...")
    with SCRYFALL_PATH.open("r") as f:
        cards = json.load(f)
    print(f"  Loaded {len(cards)} cards from Scryfall")

    lookup = {}
    for card in cards:
        name = card.get("name", "")
        normalized = format_card_name(name)
        if normalized:
            lookup[normalized] = card

        # For double-faced cards like "Delver of Secrets // Insectile Aberration",
        # also index by just the front face name
        if "//" in name:
            front_face = name.split("//")[0].strip()
            front_normalized = format_card_name(front_face)
            if front_normalized and front_normalized not in lookup:
                lookup[front_normalized] = card

    print(f"  Built lookup with {len(lookup)} entries")
    return lookup


def load_seen_cards() -> list[tuple[str, int]]:
    """
    Load all cards from seen_cards.txt with their frequency counts.

    Deduplicates by card name, keeping the highest count for each card.
    Returns sorted by frequency descending.
    """
    card_counts: dict[str, int] = {}
    with SEEN_CARDS_FILE.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            name = parts[0].strip()
            count = int(parts[1].strip())
            if name not in card_counts or count > card_counts[name]:
                card_counts[name] = count

    return sorted(card_counts.items(), key=lambda x: x[1], reverse=True)


def build_metadata():
    """Build the card metadata table by joining EDHREC and Scryfall data."""
    if not SCRYFALL_PATH.exists():
        print(f"ERROR: Scryfall data not found at {SCRYFALL_PATH}")
        print("Run download_scryfall.py first.")
        return

    scryfall_lookup = build_scryfall_lookup()
    seen_cards = load_seen_cards()
    print(f"Loaded {len(seen_cards)} cards from {SEEN_CARDS_FILE}")

    metadata = []
    matched = 0
    unmatched = 0
    unmatched_names = []

    for card_name, frequency in seen_cards:
        scryfall_card = scryfall_lookup.get(card_name)

        entry = {
            "sanitized": card_name,
            "edhrec_frequency": frequency,
        }

        if scryfall_card:
            entry.update({
                "name": scryfall_card.get("name", card_name),
                "color_identity": scryfall_card.get("color_identity", []),
                "cmc": scryfall_card.get("cmc", 0.0),
                "mana_cost": scryfall_card.get("mana_cost", ""),
                "type_line": scryfall_card.get("type_line", ""),
                "keywords": scryfall_card.get("keywords", []),
                "oracle_text": scryfall_card.get("oracle_text", ""),
                "rarity": scryfall_card.get("rarity", ""),
                "scryfall_id": scryfall_card.get("id", ""),
            })
            matched += 1
        else:
            entry.update({
                "name": card_name,
                "color_identity": [],
                "cmc": 0.0,
                "mana_cost": "",
                "type_line": "",
                "keywords": [],
                "oracle_text": "",
                "rarity": "",
                "scryfall_id": "",
            })
            unmatched += 1
            unmatched_names.append(card_name)

        metadata.append(entry)

    # Save as JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {OUTPUT_JSON} ({OUTPUT_JSON.stat().st_size / (1024*1024):.1f} MB)")

    # Save as pickle (for fast loading in Python)
    with OUTPUT_PKL.open("wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved {OUTPUT_PKL}")

    # Summary
    print(f"\n--- Metadata Summary ---")
    print(f"Total cards: {len(metadata)}")
    print(f"Matched to Scryfall: {matched} ({matched/len(metadata)*100:.1f}%)")
    print(f"Unmatched: {unmatched} ({unmatched/len(metadata)*100:.1f}%)")

    if unmatched_names:
        print(f"\nFirst 20 unmatched cards:")
        for name in unmatched_names[:20]:
            print(f"  {name}")

    # Color identity distribution
    color_counts = {}
    for entry in metadata:
        ci = tuple(sorted(entry["color_identity"]))
        label = "".join(ci) if ci else "Colorless"
        color_counts[label] = color_counts.get(label, 0) + 1

    print(f"\nColor identity distribution (top 15):")
    for ci, count in sorted(color_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {ci}: {count}")

    # Type distribution
    type_counts = {}
    for entry in metadata:
        main_type = entry["type_line"].split("—")[0].strip().split("//")[0].strip()
        if main_type:
            type_counts[main_type] = type_counts.get(main_type, 0) + 1

    print(f"\nType distribution (top 10):")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {t}: {count}")


if __name__ == "__main__":
    build_metadata()
