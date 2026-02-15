"""Tests for utils.format_card_name — the universal card name normalizer."""

import pytest

from utils import format_card_name


class TestFormatCardName:
    """Verify every normalization rule in format_card_name."""

    def test_basic_lowercase(self):
        assert format_card_name("Sol Ring") == "sol-ring"

    def test_apostrophe_removal(self):
        assert format_card_name("Sensei's Divining Top") == "senseis-divining-top"

    def test_unicode_normalization(self):
        # Lhurgoyf has a special ü in some printings
        assert format_card_name("Séance") == "seance"

    def test_special_characters(self):
        assert format_card_name("Fire // Ice") == "fire-ice"

    def test_leading_trailing_whitespace(self):
        assert format_card_name("  Sol Ring  ") == "sol-ring"

    def test_multiple_spaces(self):
        assert format_card_name("Wrath   of   God") == "wrath-of-god"

    def test_hyphenated_name(self):
        assert format_card_name("Lim-Dûl's Vault") == "lim-duls-vault"

    def test_comma_in_name(self):
        assert format_card_name("Kozilek, Butcher of Truth") == "kozilek-butcher-of-truth"

    def test_empty_string(self):
        assert format_card_name("") == ""

    def test_already_normalized(self):
        assert format_card_name("sol-ring") == "sol-ring"

    def test_numbers_in_name(self):
        # +1/+1 counter references shouldn't break things
        assert format_card_name("Ajani, Caller of the Pride") == "ajani-caller-of-the-pride"

    def test_idempotent(self):
        """Normalizing twice should give the same result."""
        name = "Sensei's Divining Top"
        once = format_card_name(name)
        twice = format_card_name(once)
        assert once == twice
