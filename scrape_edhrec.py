"""
Bulk download EDHREC synergy data using the JSON API.

Reads the top N cards from seen_cards.txt (sorted by frequency), downloads
each card's JSON data, and saves the raw response to cards_json/{card-name}.json.

Features:
- Resume capability: skips cards whose JSON file is fresh enough on disk
- Staleness check: re-downloads files older than --max-age days
- Configurable rate limiting (default: 1 request per second)
- Progress tracking with ETA
- Graceful Ctrl+C handling (data saved after each card is safe)

Usage:
    python scrape_edhrec.py --top 10 --delay 0.5    # Quick test run (10 cards)
    python scrape_edhrec.py --top 100                # Small run
    python scrape_edhrec.py --top 5000               # Full run (default)
    python scrape_edhrec.py --max-age 3              # Re-download files older than 3 days
    python scrape_edhrec.py --force                   # Re-download everything
"""

import argparse
import json
import time
from pathlib import Path

from download_card import download_card
from utils import format_card_name

CARDS_JSON_DIR = Path("cards_json")
SEEN_CARDS_FILE = Path("seen_cards.txt")
FAILED_DOWNLOADS_FILE = Path("failed_downloads.txt")

DEFAULT_TOP_N = 5000
DEFAULT_DELAY = 1.0  # seconds between requests
DEFAULT_MAX_AGE_DAYS = 14  # re-download cards older than this


def file_age_days(path: Path) -> float:
    """Return how many days old a file is based on its mtime."""
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds / 86400


def load_top_cards(seen_cards_path: Path, top_n: int) -> list[tuple[str, int]]:
    """
    Load the top N cards from seen_cards.txt, sorted by frequency (descending).

    Deduplicates by card name, keeping the highest count for each card.
    Returns a list of (card_name, count) tuples.
    """
    card_counts: dict[str, int] = {}
    with seen_cards_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            card_name = parts[0].strip()
            count = int(parts[1].strip())
            # Keep the highest count for each card
            if card_name not in card_counts or count > card_counts[card_name]:
                card_counts[card_name] = count

    cards = sorted(card_counts.items(), key=lambda x: x[1], reverse=True)
    return cards[:top_n]


def is_fresh(card_name: str, max_age_days: float) -> bool:
    """Check if a card's JSON file exists on disk and is fresh enough."""
    path = CARDS_JSON_DIR / f"{card_name}.json"
    if not path.exists():
        return False
    if path.stat().st_size == 0:
        return False  # empty files are invalid
    return file_age_days(path) < max_age_days


def download_all(top_n: int, delay: float, max_age_days: float, force: bool = False):
    """Download the top N cards from EDHREC's JSON API."""
    CARDS_JSON_DIR.mkdir(parents=True, exist_ok=True)

    if not SEEN_CARDS_FILE.exists():
        print(f"ERROR: {SEEN_CARDS_FILE} not found. Cannot determine which cards to download.")
        return

    # Load the card list
    cards = load_top_cards(SEEN_CARDS_FILE, top_n)
    print(f"Loaded {len(cards)} cards from {SEEN_CARDS_FILE} (top {top_n} by frequency)")

    if not cards:
        print("No cards to download.")
        return

    # Determine which cards need downloading
    if force:
        to_download = cards
        fresh_count = 0
        stale_count = 0
    else:
        to_download = []
        fresh_count = 0
        stale_count = 0
        for name, count in cards:
            path = CARDS_JSON_DIR / f"{name}.json"
            if not path.exists() or path.stat().st_size == 0:
                to_download.append((name, count))
            elif file_age_days(path) >= max_age_days:
                to_download.append((name, count))
                stale_count += 1
            else:
                fresh_count += 1

    print(f"Fresh (reusing): {fresh_count}")
    if stale_count > 0:
        print(f"Stale (>= {max_age_days:.0f}d, re-downloading): {stale_count}")
    print(f"Missing: {len(to_download) - stale_count}")
    print(f"Total to download: {len(to_download)}")

    if not to_download:
        print("All cards are fresh! Nothing to download.")
        return

    # Estimate time
    est_minutes = len(to_download) * delay / 60
    print(f"Estimated time: {est_minutes:.0f} minutes ({est_minutes/60:.1f} hours) at {delay}s delay")
    print()

    failed_downloads = []
    downloaded_count = 0
    start_time = time.time()

    for i, (card_name, freq) in enumerate(to_download):
        elapsed = time.time() - start_time
        if downloaded_count > 0:
            rate = downloaded_count / elapsed
            remaining = (len(to_download) - i) / rate
            eta_str = f"ETA: {remaining/60:.0f}m"
        else:
            eta_str = "ETA: calculating..."

        print(f"[{i+1}/{len(to_download)}] Downloading: {card_name} (freq: {freq}) {eta_str}")

        data = download_card(card_name)

        if not data:
            failed_downloads.append(card_name)
            print(f"  FAILED: {card_name}")
        else:
            # Save raw JSON
            output_path = CARDS_JSON_DIR / f"{card_name}.json"
            with output_path.open("w") as f:
                json.dump(data, f)
            downloaded_count += 1

        # Rate limit (skip delay on last card)
        if i < len(to_download) - 1:
            time.sleep(delay)

    # Summary
    elapsed_total = time.time() - start_time
    print()
    print(f"Download complete!")
    print(f"  Downloaded: {downloaded_count}")
    print(f"  Failed: {len(failed_downloads)}")
    print(f"  Total time: {elapsed_total/60:.1f} minutes")

    # Save failed downloads
    if failed_downloads:
        with FAILED_DOWNLOADS_FILE.open("w") as f:
            for name in failed_downloads:
                f.write(f"{name}\n")
        print(f"  Failed downloads saved to {FAILED_DOWNLOADS_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Download EDHREC card synergy data (JSON API)")
    parser.add_argument(
        "--top", type=int, default=DEFAULT_TOP_N,
        help=f"Number of top cards to download (default: {DEFAULT_TOP_N})"
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"Delay between requests in seconds (default: {DEFAULT_DELAY})"
    )
    parser.add_argument(
        "--max-age", type=float, default=DEFAULT_MAX_AGE_DAYS,
        help=f"Max age in days before re-downloading a card (default: {DEFAULT_MAX_AGE_DAYS})"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download of all cards, ignoring local cache"
    )
    args = parser.parse_args()

    print(f"=== EDHREC JSON Scraper ===")
    print(f"Top N: {args.top}")
    print(f"Delay: {args.delay}s")
    print(f"Max age: {args.max_age}d")
    if args.force:
        print(f"FORCE: re-downloading everything")
    print()

    try:
        download_all(args.top, args.delay, args.max_age, args.force)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Data saved so far is safe.")
        print("Re-run to resume from where you left off.")


if __name__ == "__main__":
    main()
