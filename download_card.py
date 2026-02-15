"""
Download a single card's synergy data from the EDHREC JSON API.

Uses the undocumented endpoint:
    https://json.edhrec.com/pages/cards/{card-name}.json

Returns the raw JSON response as a Python dict.
"""

import time

import requests

from utils import format_card_name

EDHREC_JSON_URL = "https://json.edhrec.com/pages/cards/{card_name}.json"
USER_AGENT = "Opticube/0.1 (personal noncommercial MTG cube optimizer)"
MAX_RETRIES = 5


def download_card(card_name: str) -> dict:
    """
    Download a single card's EDHREC JSON data.

    Args:
        card_name: The card name (will be formatted to a URL slug).

    Returns:
        The parsed JSON response as a dict, or {} on failure.
    """
    slug = format_card_name(card_name)
    url = EDHREC_JSON_URL.format(card_name=slug)
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    backoff_time = 1

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                print(f"  Card not found on EDHREC: {slug} (404)")
                return {}
            else:
                raise requests.exceptions.RequestException(
                    f"HTTP {response.status_code} for {slug}"
                )
        except requests.exceptions.RequestException as e:
            print(f"  Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(backoff_time)
                backoff_time *= 2
            else:
                print(f"  Failed to download {slug} after {MAX_RETRIES} attempts.")
                return {}

    return {}


if __name__ == "__main__":
    import json

    data = download_card("Lightning Bolt")
    if data:
        print(json.dumps(data, indent=2)[:2000])
        print(f"\n... (total keys: {list(data.keys())})")
    else:
        print("Download failed.")
