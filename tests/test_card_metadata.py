"""
Tests for card_metadata.py — theme detection, property building, Scryfall loading.

Uses synthetic/mock data from conftest.py.
"""

import json
import tempfile
from pathlib import Path

import pytest

from card_metadata import (
    THEME_PATTERNS,
    detect_card_themes,
    build_card_properties,
    get_theme_card_sets,
    load_scryfall_data,
)
from tests.conftest import CARD_NAMES_10


# ===================================================================
# detect_card_themes
# ===================================================================

class TestDetectCardThemes:
    """Regex-based theme detection on oracle text."""

    def test_lifegain_detected(self):
        text = "Whenever a creature enters the battlefield, you gain 2 life."
        themes = detect_card_themes(text)
        assert "lifegain" in themes

    def test_tokens_detected(self):
        text = "Create two 1/1 white Soldier creature tokens."
        themes = detect_card_themes(text)
        assert "tokens" in themes

    def test_sacrifice_detected(self):
        text = "Sacrifice a creature: Draw a card."
        themes = detect_card_themes(text)
        assert "sacrifice" in themes

    def test_graveyard_detected(self):
        text = "Return target creature card from your graveyard to the battlefield."
        themes = detect_card_themes(text)
        assert "graveyard" in themes

    def test_spellslinger_detected(self):
        text = "Whenever you cast an instant or sorcery spell, draw a card."
        themes = detect_card_themes(text)
        assert "spellslinger" in themes

    def test_counters_plus1_detected(self):
        text = "Put a +1/+1 counter on target creature."
        themes = detect_card_themes(text)
        assert "counters_plus1" in themes

    def test_ramp_detected(self):
        text = "Search your library for a basic land card and put it onto the battlefield."
        themes = detect_card_themes(text)
        assert "ramp" in themes

    def test_draw_detected(self):
        text = "Draw three cards."
        themes = detect_card_themes(text)
        assert "draw" in themes

    def test_flicker_detected(self):
        text = "Exile target creature, then return it to the battlefield."
        themes = detect_card_themes(text)
        assert "flicker" in themes

    def test_control_counter_target(self):
        text = "Counter target spell."
        themes = detect_card_themes(text)
        assert "control" in themes

    def test_aggro_haste(self):
        themes = detect_card_themes("", type_line="Creature — Goblin")
        # Just type line without haste keywords → no aggro
        assert "aggro" not in themes

        themes2 = detect_card_themes("Haste")
        assert "aggro" in themes2

    def test_no_themes_for_vanilla(self):
        text = "3/3"
        themes = detect_card_themes(text)
        assert len(themes) == 0

    def test_multiple_themes_detected(self):
        text = "Sacrifice a creature: Create two 1/1 tokens and draw a card."
        themes = detect_card_themes(text)
        assert "sacrifice" in themes
        assert "tokens" in themes
        assert "draw" in themes

    def test_type_line_contributes(self):
        themes = detect_card_themes("", type_line="Artifact — Equipment")
        assert "artifacts" in themes

    def test_case_insensitive(self):
        themes = detect_card_themes("WHENEVER YOU CAST AN INSTANT OR SORCERY SPELL, DRAW A CARD.")
        assert "spellslinger" in themes
        assert "draw" in themes


# ===================================================================
# build_card_properties
# ===================================================================

class TestBuildCardProperties:
    """Building per-card property dicts from Scryfall data."""

    def test_all_property_keys_present(self, mock_scryfall_data):
        props = build_card_properties(CARD_NAMES_10, mock_scryfall_data)
        expected_keys = {'color_identity', 'cmc', 'type_line', 'keywords', 'oracle_text', 'themes'}
        assert set(props.keys()) == expected_keys

    def test_correct_color_identity(self, mock_scryfall_data):
        props = build_card_properties(CARD_NAMES_10, mock_scryfall_data)
        # Sol Ring is colorless
        assert props['color_identity'][0] == []
        # Lightning Bolt is Red
        assert props['color_identity'][1] == ['R']
        # Counterspell is Blue
        assert props['color_identity'][3] == ['U']

    def test_correct_cmc(self, mock_scryfall_data):
        props = build_card_properties(CARD_NAMES_10, mock_scryfall_data)
        # Sol Ring = 1, Counterspell = 2, Wrath of God = 4, Cultivate = 3
        assert props['cmc'][0] == 1.0
        assert props['cmc'][3] == 2.0
        assert props['cmc'][7] == 4.0
        assert props['cmc'][9] == 3.0

    def test_correct_type_line(self, mock_scryfall_data):
        props = build_card_properties(CARD_NAMES_10, mock_scryfall_data)
        assert "Artifact" in props['type_line'][0]
        assert "Instant" in props['type_line'][1]
        assert "Creature" in props['type_line'][2]
        assert "Sorcery" in props['type_line'][7]

    def test_theme_detection_integrated(self, mock_scryfall_data):
        props = build_card_properties(CARD_NAMES_10, mock_scryfall_data)
        # Brainstorm's oracle text: "Draw three cards..." -> draw theme
        assert 'draw' in props['themes'][8]
        # Cultivate: "Search your library for...land" -> ramp theme
        assert 'ramp' in props['themes'][9]
        # Counterspell: "Counter target spell." -> control theme
        assert 'control' in props['themes'][3]

    def test_all_10_cards_matched(self, mock_scryfall_data):
        props = build_card_properties(CARD_NAMES_10, mock_scryfall_data)
        assert len(props['cmc']) == 10

    def test_unmatched_card_skipped(self, mock_scryfall_data):
        """If a card name doesn't exist in Scryfall, it's just skipped."""
        names_with_unknown = list(CARD_NAMES_10) + ["Totally Fake Card"]
        props = build_card_properties(names_with_unknown, mock_scryfall_data)
        # Only 10 of 11 matched
        assert len(props['cmc']) == 10
        assert 10 not in props['cmc']


# ===================================================================
# get_theme_card_sets
# ===================================================================

class TestGetThemeCardSets:
    """Build theme -> card index mapping."""

    def test_returns_all_themes(self, mock_scryfall_data):
        props = build_card_properties(CARD_NAMES_10, mock_scryfall_data)
        theme_sets = get_theme_card_sets(props, len(CARD_NAMES_10))
        assert set(theme_sets.keys()) == set(THEME_PATTERNS.keys())

    def test_draw_cards_correct(self, mock_scryfall_data):
        props = build_card_properties(CARD_NAMES_10, mock_scryfall_data)
        theme_sets = get_theme_card_sets(props, len(CARD_NAMES_10))
        # Brainstorm (idx 8) draws cards
        assert 8 in theme_sets['draw']

    def test_ramp_cards_correct(self, mock_scryfall_data):
        props = build_card_properties(CARD_NAMES_10, mock_scryfall_data)
        theme_sets = get_theme_card_sets(props, len(CARD_NAMES_10))
        # Cultivate (idx 9) is ramp
        assert 9 in theme_sets['ramp']

    def test_empty_themes_are_empty_sets(self, mock_scryfall_data):
        props = build_card_properties(CARD_NAMES_10, mock_scryfall_data)
        theme_sets = get_theme_card_sets(props, len(CARD_NAMES_10))
        # Some themes might have no cards (e.g., flicker)
        for theme, card_set in theme_sets.items():
            assert isinstance(card_set, set)


# ===================================================================
# load_scryfall_data
# ===================================================================

class TestLoadScryfallData:
    """Test loading from a mock Scryfall JSON file."""

    def test_loads_and_normalizes(self):
        """Write a small JSON, load it, verify normalization."""
        cards = [
            {"name": "Sol Ring", "cmc": 1.0, "color_identity": []},
            {"name": "Jace, the Mind Sculptor", "cmc": 4.0, "color_identity": ["U"]},
            {"name": "Fire // Ice", "cmc": 2.0, "color_identity": ["U", "R"]},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cards, f)
            tmp_path = f.name

        lookup = load_scryfall_data(tmp_path)

        assert "sol-ring" in lookup
        assert "jace-the-mind-sculptor" in lookup
        # Double-faced card indexed by front face
        assert "fire" in lookup
        assert len(lookup) == 3

    def test_handles_double_faced(self):
        cards = [
            {"name": "Delver of Secrets // Insectile Aberration", "cmc": 1.0},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cards, f)
            tmp_path = f.name

        lookup = load_scryfall_data(tmp_path)
        assert "delver-of-secrets" in lookup
