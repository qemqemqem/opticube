"""
Card Metadata Module

Loads Scryfall bulk data and builds per-card property tables for use as
constraints in the optimizer (color balance, mana curve, card types, themes).

Also provides keyword/theme detection from oracle text for archetype quotas.

Usage:
    from card_metadata import load_scryfall_data, build_card_properties

    scryfall = load_scryfall_data('data/scryfall_oracle_cards.json')
    props = build_card_properties(card_names, scryfall)
    # props['color_identity'][42] -> ['R', 'G']
    # props['cmc'][42] -> 3.0
    # props['themes'][42] -> {'ramp', 'tokens'}
"""

import json
import re
from pathlib import Path

from utils import format_card_name


# ---------------------------------------------------------------------------
# Theme detection patterns (oracle text regex matching)
# ---------------------------------------------------------------------------

THEME_PATTERNS = {
    'counters_plus1': [r'\+1/\+1 counter'],
    'counters_minus1': [r'-1/-1 counter'],
    'lifegain': [r'gain.*life', r'life.*gain', r'lifelink'],
    'graveyard': [r'graveyard', r'return.*from.*graveyard', r'mill'],
    'tokens': [r'create.*token', r'token.*creature'],
    'sacrifice': [r'sacrifice', r'when.*dies'],
    'spellslinger': [r'whenever you cast.*(instant|sorcery)', r'magecraft'],
    'artifacts': [r'artifact.*enter', r'equip\b', r'equipment'],
    'enchantments': [r'enchantment.*enter', r'aura', r'constellation'],
    'tribal': [r'each.*you control get', r'other.*you control get'],
    'ramp': [r'search.*library.*land', r'add \{', r'add.*mana'],
    'draw': [r'draw.*card', r'whenever.*draw'],
    'flicker': [r'exile.*return.*to the battlefield', r'blink'],
    'voltron': [r'equipped creature', r'enchanted creature get'],
    'control': [r'counter target', r'destroy target', r'exile target'],
    'aggro': [r'haste', r'first strike', r'double strike', r'menace'],
}


# ---------------------------------------------------------------------------
# Scryfall data loading
# ---------------------------------------------------------------------------

def load_scryfall_data(path):
    """
    Load Scryfall oracle cards bulk data.

    Download from: https://api.scryfall.com/bulk-data/oracle-cards
    (follow the download_uri in the response)

    Args:
        path: Path to the Scryfall oracle_cards JSON file

    Returns:
        Dict mapping normalized card name -> card data dict
    """
    path = Path(path)
    print(f"  Loading Scryfall data from {path}...")

    with path.open('r', encoding='utf-8') as f:
        cards = json.load(f)

    lookup = {}
    for card in cards:
        name = card.get('name', '')
        # Handle double-faced cards: "Front // Back" -> index by front face
        if ' // ' in name:
            name = name.split(' // ')[0]
        normalized = format_card_name(name)
        lookup[normalized] = card

    print(f"  Loaded {len(lookup)} unique cards from Scryfall")
    return lookup


# ---------------------------------------------------------------------------
# Card properties builder
# ---------------------------------------------------------------------------

def build_card_properties(card_names, scryfall_data):
    """
    Build per-card property dicts for optimizer constraints.

    Joins EDHREC card names (from the synergy matrix) with Scryfall metadata.

    Args:
        card_names: List of card names (indexed by matrix position)
        scryfall_data: Dict from load_scryfall_data()

    Returns:
        Dict with keys: 'color_identity', 'cmc', 'type_line', 'keywords',
                        'oracle_text', 'themes'
        Each value is a dict mapping card_index -> property_value
    """
    properties = {
        'color_identity': {},
        'cmc': {},
        'type_line': {},
        'keywords': {},
        'oracle_text': {},
        'themes': {},
    }

    matched = 0
    for idx, name in enumerate(card_names):
        normalized = format_card_name(name)
        card = scryfall_data.get(normalized)
        if card is None:
            continue

        matched += 1
        properties['color_identity'][idx] = card.get('color_identity', [])
        properties['cmc'][idx] = card.get('cmc', 0.0)
        properties['type_line'][idx] = card.get('type_line', '')
        properties['keywords'][idx] = card.get('keywords', [])

        oracle = card.get('oracle_text', '')
        properties['oracle_text'][idx] = oracle
        properties['themes'][idx] = detect_card_themes(
            oracle, card.get('type_line', ''),
        )

    print(f"  Matched {matched}/{len(card_names)} cards to Scryfall data")
    return properties


# ---------------------------------------------------------------------------
# Theme detection
# ---------------------------------------------------------------------------

def detect_card_themes(oracle_text, type_line=''):
    """
    Detect which themes a card belongs to based on oracle text and type line.

    Uses regex matching against THEME_PATTERNS.

    Args:
        oracle_text: Card's oracle text
        type_line: Card's type line

    Returns:
        Set of theme names this card belongs to
    """
    text = (oracle_text + ' ' + type_line).lower()
    themes = set()

    for theme_name, patterns in THEME_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                themes.add(theme_name)
                break

    return themes


def get_theme_card_sets(card_properties, n_cards):
    """
    Build a dict mapping theme_name -> set of card indices for that theme.

    Useful for adding theme quota constraints to the optimizer.

    Args:
        card_properties: Output of build_card_properties()
        n_cards: Total number of cards

    Returns:
        Dict of theme_name -> set of card indices
    """
    theme_sets = {theme: set() for theme in THEME_PATTERNS}

    themes_dict = card_properties.get('themes', {})
    for idx, card_themes in themes_dict.items():
        for theme in card_themes:
            if theme in theme_sets:
                theme_sets[theme].add(idx)

    return theme_sets


def summarize_properties(card_properties, card_names):
    """
    Print a summary of the card properties for inspection.

    Args:
        card_properties: Output of build_card_properties()
        card_names: List of card names
    """
    n_total = len(card_names)
    n_matched = len(card_properties.get('cmc', {}))

    print(f"\n  Card Metadata Summary ({n_matched}/{n_total} matched)")
    print(f"  {'='*50}")

    # Color distribution
    if 'color_identity' in card_properties:
        color_counts = {'W': 0, 'U': 0, 'B': 0, 'R': 0, 'G': 0, 'colorless': 0}
        for idx, colors in card_properties['color_identity'].items():
            if not colors:
                color_counts['colorless'] += 1
            for c in colors:
                if c in color_counts:
                    color_counts[c] += 1
        print(f"  Colors: {color_counts}")

    # CMC distribution
    if 'cmc' in card_properties:
        cmcs = list(card_properties['cmc'].values())
        if cmcs:
            print(f"  CMC: min={min(cmcs):.0f}, max={max(cmcs):.0f}, "
                  f"mean={sum(cmcs)/len(cmcs):.1f}")
            brackets = {
                '0-1': sum(1 for c in cmcs if c <= 1),
                '2': sum(1 for c in cmcs if c == 2),
                '3': sum(1 for c in cmcs if c == 3),
                '4': sum(1 for c in cmcs if c == 4),
                '5': sum(1 for c in cmcs if c == 5),
                '6+': sum(1 for c in cmcs if c >= 6),
            }
            print(f"  CMC brackets: {brackets}")

    # Type distribution
    if 'type_line' in card_properties:
        type_counts = {
            'Creature': 0, 'Instant': 0, 'Sorcery': 0,
            'Enchantment': 0, 'Artifact': 0, 'Planeswalker': 0, 'Land': 0,
        }
        for idx, type_line in card_properties['type_line'].items():
            for card_type in type_counts:
                if card_type in type_line:
                    type_counts[card_type] += 1
        print(f"  Types: {type_counts}")

    # Theme distribution
    if 'themes' in card_properties:
        theme_counts = {}
        for idx, themes in card_properties['themes'].items():
            for theme in themes:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
        sorted_themes = sorted(theme_counts.items(), key=lambda x: -x[1])
        print(f"  Themes (top 10):")
        for theme, count in sorted_themes[:10]:
            print(f"    {theme}: {count}")
