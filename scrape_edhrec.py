import heapq
from pathlib import Path
import os
from collections import deque, defaultdict
import re

def initialize_files_and_directories():
    # Define file and directory paths
    seen_cards_file = Path('seen_cards.txt')
    downloaded_cards_file = Path('downloaded_cards.txt')
    cards_dir = Path('cards')

    # Ensure the cards directory exists
    cards_dir.mkdir(exist_ok=True)

    # Create the files if they don't exist
    seen_cards_file.touch(exist_ok=True)
    downloaded_cards_file.touch(exist_ok=True)

    return seen_cards_file, downloaded_cards_file, cards_dir

# Function to load cards from a file into a set
def load_cards(file_path: Path) -> set:
    with file_path.open('r') as f:
        return set(line.strip() for line in f)

# Function to save a card to a file
def save_card(file_path: Path, card_name: str) -> None:
    with file_path.open('a') as f:
        f.write(f"{card_name}\n")


# Function to load cards from a file into a dictionary with counts
def load_cards_with_counts(file_path: Path) -> defaultdict:
    cards_with_counts = defaultdict(int)  # default value of int is 0
    with file_path.open('r') as f:
        for line in f:
            card_name, count = line.strip().split(',')
            cards_with_counts[card_name] = int(count)
    return cards_with_counts

# Function to save a card with count to a file
def save_card_with_count(file_path: Path, card_name: str, count: int) -> None:
    with file_path.open('a') as f:
        f.write(f"{card_name},{count}\n")

# Placeholder function for downloading a card
def download_card(card_name: str) -> None:
    # TODO: Implement this function
    pass

# Function to format card names
def format_card_name(card_name: str) -> str:
    # Convert to lower case
    card_name = card_name.lower()
    # Replace non-alphanumeric characters with underscores
    card_name = re.sub(r'\W+', '_', card_name)
    # Remove leading/trailing underscores
    card_name = card_name.strip('_')
    return card_name

def initialize_priority_queue(seen_cards, downloaded_cards):
    download_next = []
    for card, count in seen_cards.items():
        if card not in downloaded_cards:
            heapq.heappush(download_next, (-count, card))
    return download_next

def process_download_queue(download_next, seen_cards, downloaded_cards, seen_cards_file, downloaded_cards_file, cards_dir):
    processed_count = 0
    while download_next:
        _, card_name = heapq.heappop(download_next)

        if card_name in downloaded_cards:
            continue

        download_card(card_name)
        seen_cards[card_name] += 1
        downloaded_cards.add(card_name)

        save_card_with_count(seen_cards_file, card_name, seen_cards[card_name])
        save_card(downloaded_cards_file, card_name)

        card_file = cards_dir / f"{card_name}.txt"
        card_file.touch()
        processed_count += 1

    return processed_count

def main():
    seen_cards_file, downloaded_cards_file, cards_dir = initialize_files_and_directories()

    seen_cards = load_cards_with_counts(seen_cards_file)
    downloaded_cards = load_cards(downloaded_cards_file)

    download_next = initialize_priority_queue(seen_cards, downloaded_cards)
    processed_count = 0
    try:
        processed_count = process_download_queue(download_next, seen_cards, downloaded_cards, seen_cards_file, downloaded_cards_file, cards_dir)
    except KeyboardInterrupt:
        print("Process was interrupted by user.")

    print(f"{processed_count} cards processed.")
    print(f"Seen cards file size: {seen_cards_file.stat().st_size} bytes")
    print(f"Downloaded cards file size: {downloaded_cards_file.stat().st_size} bytes")

if __name__ == "__main__":
    main()
