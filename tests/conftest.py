"""
Shared synthetic test fixtures for the Opticube test suite.

All data here is hand-crafted so we know the expected answers.
"""

import numpy as np
import pytest
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Card names
# ---------------------------------------------------------------------------

CARD_NAMES_10 = [
    "Sol Ring",
    "Lightning Bolt",
    "Llanowar Elves",
    "Counterspell",
    "Swords to Plowshares",
    "Dark Ritual",
    "Birds of Paradise",
    "Wrath of God",
    "Brainstorm",
    "Cultivate",
]


# ---------------------------------------------------------------------------
# Small synergy matrices (10 x 10)
# ---------------------------------------------------------------------------

@pytest.fixture
def card_names_10():
    """List of 10 recognizable card names."""
    return list(CARD_NAMES_10)


@pytest.fixture
def symmetric_synergy_10():
    """
    10x10 symmetric sparse synergy matrix with known structure.

    Block structure:
      - Cards 0-4 form a tightly-connected group (high synergy among each other)
      - Cards 5-9 form a second group
      - Cross-group synergy is weak
    This lets us verify that the optimizer picks one of the two blocks.
    """
    n = 10
    data = np.zeros((n, n))

    # Block A: cards 0-4 (strong synergy = 5.0)
    for i in range(5):
        for j in range(i + 1, 5):
            data[i, j] = 5.0
            data[j, i] = 5.0

    # Block B: cards 5-9 (moderate synergy = 3.0)
    for i in range(5, 10):
        for j in range(i + 1, 10):
            data[i, j] = 3.0
            data[j, i] = 3.0

    # Cross-block: weak synergy = 0.5
    for i in range(5):
        for j in range(5, 10):
            data[i, j] = 0.5
            data[j, i] = 0.5

    return sp.csr_matrix(data)


@pytest.fixture
def asymmetric_synergy_10():
    """
    10x10 asymmetric sparse synergy matrix for testing symmetrize_matrix.

    Hand-placed asymmetries:
      - (0,1)=4.0, (1,0)=6.0  -> should average to 5.0
      - (2,3)=3.0, (3,2)=0.0  -> only one direction, should keep 3.0
      - (7,8)=0.0, (8,7)=2.0  -> only one direction, should keep 2.0
    """
    n = 10
    data = np.zeros((n, n))

    # Both directions (should average)
    data[0, 1] = 4.0
    data[1, 0] = 6.0

    data[3, 4] = 2.0
    data[4, 3] = 8.0

    # One direction only
    data[2, 3] = 3.0
    # (3,2) deliberately left at 0

    data[8, 7] = 2.0
    # (7,8) deliberately left at 0

    # A fully symmetric pair for sanity
    data[5, 6] = 4.0
    data[6, 5] = 4.0

    return sp.csr_matrix(data)


@pytest.fixture
def synergy_with_clear_optimum():
    """
    6x6 matrix where picking cards {0,1,2} is clearly optimal for set_size=3.

    Cards 0,1,2 have pairwise synergy of 10.0.
    Cards 3,4,5 have pairwise synergy of 1.0.
    Cross-group synergy is 0.0.
    Optimal 3-card set = {0,1,2} with score = 30.0 (upper triangle sum).
    """
    n = 6
    data = np.zeros((n, n))

    for i in range(3):
        for j in range(i + 1, 3):
            data[i, j] = 10.0
            data[j, i] = 10.0

    for i in range(3, 6):
        for j in range(i + 1, 6):
            data[i, j] = 1.0
            data[j, i] = 1.0

    return sp.csr_matrix(data)


# ---------------------------------------------------------------------------
# Larger synthetic matrix (50 cards) for spectral / greedy tests
# ---------------------------------------------------------------------------

@pytest.fixture
def synergy_50():
    """
    50x50 symmetric matrix with 5 clusters of 10 cards each.

    Intra-cluster synergy = 4.0, inter-cluster = 0.2.
    Useful for testing spectral clustering and greedy at moderate scale.
    """
    n = 50
    data = np.zeros((n, n))
    cluster_size = 10

    for cluster in range(5):
        start = cluster * cluster_size
        end = start + cluster_size
        for i in range(start, end):
            for j in range(i + 1, end):
                data[i, j] = 4.0
                data[j, i] = 4.0

    # Weak inter-cluster connections
    rng = np.random.RandomState(42)
    for i in range(n):
        for j in range(i + 1, n):
            if data[i, j] == 0:
                if rng.random() < 0.1:  # sparse cross-links
                    val = 0.2
                    data[i, j] = val
                    data[j, i] = val

    return sp.csr_matrix(data)


@pytest.fixture
def card_names_50():
    """50 synthetic card names."""
    return [f"Card_{i:03d}" for i in range(50)]


# ---------------------------------------------------------------------------
# Mock Scryfall data
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_scryfall_data():
    """
    Fake Scryfall-like lookup dict for 10 cards matching CARD_NAMES_10.

    Provides color_identity, cmc, type_line, oracle_text, keywords
    so we can test build_card_properties and detect_card_themes.
    """
    from utils import format_card_name

    cards = [
        {
            "name": "Sol Ring",
            "color_identity": [],
            "cmc": 1.0,
            "type_line": "Artifact",
            "oracle_text": "{T}: Add {C}{C}.",
            "keywords": [],
        },
        {
            "name": "Lightning Bolt",
            "color_identity": ["R"],
            "cmc": 1.0,
            "type_line": "Instant",
            "oracle_text": "Lightning Bolt deals 3 damage to any target.",
            "keywords": [],
        },
        {
            "name": "Llanowar Elves",
            "color_identity": ["G"],
            "cmc": 1.0,
            "type_line": "Creature — Elf Druid",
            "oracle_text": "{T}: Add {G}.",
            "keywords": [],
        },
        {
            "name": "Counterspell",
            "color_identity": ["U"],
            "cmc": 2.0,
            "type_line": "Instant",
            "oracle_text": "Counter target spell.",
            "keywords": [],
        },
        {
            "name": "Swords to Plowshares",
            "color_identity": ["W"],
            "cmc": 1.0,
            "type_line": "Instant",
            "oracle_text": "Exile target creature. Its controller gains life equal to its power.",
            "keywords": [],
        },
        {
            "name": "Dark Ritual",
            "color_identity": ["B"],
            "cmc": 1.0,
            "type_line": "Instant",
            "oracle_text": "Add {B}{B}{B}.",
            "keywords": [],
        },
        {
            "name": "Birds of Paradise",
            "color_identity": ["G"],
            "cmc": 1.0,
            "type_line": "Creature — Bird",
            "oracle_text": "{T}: Add one mana of any color.",
            "keywords": ["Flying"],
        },
        {
            "name": "Wrath of God",
            "color_identity": ["W"],
            "cmc": 4.0,
            "type_line": "Sorcery",
            "oracle_text": "Destroy all creatures. They can't be regenerated.",
            "keywords": [],
        },
        {
            "name": "Brainstorm",
            "color_identity": ["U"],
            "cmc": 1.0,
            "type_line": "Instant",
            "oracle_text": "Draw three cards, then put two cards from your hand on top of your library in any order.",
            "keywords": [],
        },
        {
            "name": "Cultivate",
            "color_identity": ["G"],
            "cmc": 3.0,
            "type_line": "Sorcery",
            "oracle_text": "Search your library for up to two basic land cards, reveal those cards, put one onto the battlefield tapped and the other into your hand, then shuffle.",
            "keywords": [],
        },
    ]

    lookup = {}
    for card in cards:
        normalized = format_card_name(card["name"])
        lookup[normalized] = card

    return lookup
