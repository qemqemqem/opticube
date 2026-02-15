"""
Download Scryfall Oracle Cards bulk data.

Fetches the bulk data metadata from the Scryfall API, then downloads the
full Oracle Cards JSON file to data/scryfall_oracle_cards.json.

Reuses a local copy if it exists and is fresh enough (controlled by --max-age).
"""

import argparse
import json
import time
from pathlib import Path

import requests

BULK_DATA_URL = "https://api.scryfall.com/bulk-data/oracle-cards"
OUTPUT_PATH = Path("data/scryfall_oracle_cards.json")

USER_AGENT = "Opticube/0.1 (personal noncommercial MTG cube optimizer)"

DEFAULT_MAX_AGE_DAYS = 7


def file_age_days(path: Path) -> float:
    """Return how many days old a file is based on its mtime."""
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds / 86400


def download_scryfall_bulk(max_age_days: float = DEFAULT_MAX_AGE_DAYS, force: bool = False):
    """Download the Scryfall Oracle Cards bulk data file."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and not force:
        age = file_age_days(OUTPUT_PATH)
        size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
        if age < max_age_days:
            print(f"Scryfall data is fresh ({age:.1f} days old, max {max_age_days}d). Reusing {OUTPUT_PATH} ({size_mb:.1f} MB)")
            return
        else:
            print(f"Scryfall data is stale ({age:.1f} days old, max {max_age_days}d). Re-downloading...")
            OUTPUT_PATH.unlink()

    # Step 1: Get the download URI from the bulk data endpoint
    print(f"Fetching bulk data metadata from {BULK_DATA_URL}...")
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    resp = requests.get(BULK_DATA_URL, headers=headers, timeout=30)
    resp.raise_for_status()

    metadata = resp.json()
    download_uri = metadata["download_uri"]
    print(f"Download URI: {download_uri}")
    print(f"Updated at: {metadata.get('updated_at', 'unknown')}")

    # Step 2: Stream-download the actual data file
    print("Downloading Oracle Cards data (this may take a minute)...")
    resp = requests.get(download_uri, headers=headers, stream=True, timeout=300)
    resp.raise_for_status()

    total_size = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(OUTPUT_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded / total_size * 100
                print(f"\r  {downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB ({pct:.0f}%)", end="", flush=True)
            else:
                print(f"\r  {downloaded / (1024*1024):.1f} MB downloaded", end="", flush=True)

    print()  # newline after progress

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved to {OUTPUT_PATH} ({size_mb:.1f} MB)")

    # Step 3: Quick validation -- load and count cards
    validate_scryfall()


def validate_scryfall():
    """Load and print basic stats about the Scryfall data."""
    print("Validating...")
    with open(OUTPUT_PATH, "r") as f:
        cards = json.load(f)

    print(f"Total cards in Scryfall bulk data: {len(cards)}")

    type_counts = {}
    for card in cards:
        main_type = card.get("type_line", "Unknown").split("—")[0].strip()
        type_counts[main_type] = type_counts.get(main_type, 0) + 1

    print("Top card types:")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {t}: {count}")


def build_lookup_table():
    """
    Build a name-keyed lookup table from the Scryfall bulk data.
    Returns a dict mapping normalized card name -> card data.
    """
    from utils import format_card_name

    with open(OUTPUT_PATH, "r") as f:
        cards = json.load(f)

    lookup = {}
    for card in cards:
        name = card.get("name", "")
        normalized = format_card_name(name)
        if normalized:
            lookup[normalized] = card

    return lookup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Scryfall Oracle Cards bulk data")
    parser.add_argument(
        "--max-age", type=float, default=DEFAULT_MAX_AGE_DAYS,
        help=f"Max age in days before re-downloading (default: {DEFAULT_MAX_AGE_DAYS})"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download even if local file is fresh"
    )
    args = parser.parse_args()
    download_scryfall_bulk(max_age_days=args.max_age, force=args.force)
