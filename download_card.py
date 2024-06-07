import requests
import html2text

import re
from utils import format_card_name

# Example output

# [EDHREC]
# ...
# Utility_Lands
# Mana_Artifacts
# Lands
# Top Commanders (13)
# Saheeli, the Sun&#x27;s Brilliance
# 8.16% of 1029 decks 84 decks
# Dr. Madison Li
# 5.72% of 2257 decks 129 decks
# Urza, Chief Artificer
# 5.00% of 11526 decks 576 decks


def parse_card_details(markdown_text: str) -> dict:
    """
    Parses the markdown text to extract card details.
    This is a basic parser and might need adjustments based on actual markdown structure.
    
    Args:
    markdown_text (str): The markdown text of the card details.

    Returns:
    dict: A dictionary containing parsed details of the card.
    """
    lines = markdown_text.split('\n')
    card_details = {}
    previous_line = None  # To hold the potential card name

    for line in lines:
        if not line.strip():
            continue
        # Updated regex to match new potential format with synergy
        if re.match(r'\d+(\.\d+)?%\s+of\s+\d+\s+decks(\s+\+\d+% synergy)?', line):
            if previous_line:  # Ensure there is a previous line which is the card name
                card_details[previous_line] = line
        # elif "%" in line:
        #     print("Surprise unhandled line!", line)
        previous_line = line  # Update previous_line for the next iteration

    return card_details

import time

def download_card(card_name: str) -> dict:
    url = f"https://edhrec.com/cards/{format_card_name(card_name)}"
    backoff_time = 1  # Initial backoff time in seconds
    max_retries = 5  # Maximum number of retries

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)  # Set a timeout for the request
            if response.status_code == 200:
                text = html2text.html2text(response.text)
                card_details = parse_card_details(text)
                backoff_time = min(1, backoff_time / 1.2)  # Shrink a little more slowly
                return card_details
            else:
                raise requests.exceptions.RequestException(f"Failed to download card: {card_name}, Status Code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff
            else:
                print(f"Failed to download card: {card_name} after {max_retries} attempts.")
                return {}

if __name__ == "__main__":
    # Example usage
    download_card("Simulacrum Synthesizer")
