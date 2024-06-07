import heapq
from pathlib import Path
import os
from collections import deque, defaultdict

from download_card import download_card
from utils import format_card_name

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
# Function to format card names
def initialize_priority_queue(seen_cards, downloaded_cards):
    download_next = []
    for card, count in seen_cards.items():
        if card not in downloaded_cards:
            heapq.heappush(download_next, (-count, card))
    
    # If no cards are available, initialize with "Lightning Bolt"
    if not download_next:
        heapq.heappush(download_next, (-1, format_card_name("Lightning Bolt")))  # Default count as -1

    return download_next

def process_download_queue(download_next, seen_cards, downloaded_cards, seen_cards_file, downloaded_cards_file, cards_dir, failed_downloads, max_count=-1):
    processed_count = 0
    while download_next and (max_count <= 0 or processed_count < max_count):
        _, card_name = heapq.heappop(download_next)

        print(f"Downloading card: {card_name}")

        if card_name in downloaded_cards:
            continue

        card_details = download_card(card_name)
        if not card_details:
            failed_downloads.append(card_name)
            continue
        for key in card_details.keys():
            formatted_name = format_card_name(key)
            if formatted_name not in seen_cards:
                seen_cards[formatted_name] = 0  # Initialize if not already seen
            seen_cards[formatted_name] += 1  # Increment the count for the seen card
            save_card_with_count(seen_cards_file, formatted_name, seen_cards[formatted_name])

        downloaded_cards.add(card_name)
        save_card(downloaded_cards_file, card_name)

        card_file = cards_dir / f"{card_name}.txt"
        with card_file.open('w') as file:
            for key, value in card_details.items():
                file.write(f"{key}: {value}\n")

        processed_count += 1

        download_next = [item for item in initialize_priority_queue(seen_cards, downloaded_cards) if item[1] not in failed_downloads]

    return processed_count
def reset_card_data(seen_cards_file: Path, downloaded_cards_file: Path, cards_dir: Path) -> None:
    # Empty the seen_cards.txt and downloaded_cards.txt files
    seen_cards_file.write_text('')
    downloaded_cards_file.write_text('')

    # Remove all files in the cards directory
    for card_file in cards_dir.iterdir():
        card_file.unlink()

    print("Card data has been reset.")

def main():
    seen_cards_file, downloaded_cards_file, cards_dir = initialize_files_and_directories()

    # reset_card_data(seen_cards_file, downloaded_cards_file, cards_dir)  # WARNING DO NOT UNCOMMENT

    seen_cards = load_cards_with_counts(seen_cards_file)
    downloaded_cards = load_cards(downloaded_cards_file)

    download_next = initialize_priority_queue(seen_cards, downloaded_cards)
    failed_downloads = []  # Initialize the list to track failed downloads
    processed_count = 0
    try:
        processed_count = process_download_queue(download_next, seen_cards, downloaded_cards, seen_cards_file, downloaded_cards_file, cards_dir, failed_downloads, max_count=-1)
    except KeyboardInterrupt:
        print("Process was interrupted by user.")

    print(f"{processed_count} cards processed.")
    print(f"Seen cards file size: {seen_cards_file.stat().st_size} bytes")
    print(f"Downloaded cards file size: {downloaded_cards_file.stat().st_size} bytes")
    if failed_downloads:
        with open('failed_downloads.txt', 'w') as f:
            for failed_download in failed_downloads:
                f.write(f"{failed_download}\n")
        print("Failed downloads have been saved to failed_downloads.txt")

if __name__ == "__main__":
    main()
