import re
import unicodedata

def format_card_name(card_name: str) -> str:
    # Normalize Unicode characters to their closest ASCII representation
    card_name = unicodedata.normalize('NFKD', card_name).encode('ascii', 'ignore').decode('ascii')
    # Convert to lower case
    card_name = card_name.lower().strip()
    # Remove all apostrophes
    card_name = card_name.replace("'", "")
    # Replace non-alphanumeric characters with underscores
    card_name = re.sub(r'\W+', '-', card_name)
    # Remove leading/trailing underscores
    card_name = card_name.strip('-')
    return card_name