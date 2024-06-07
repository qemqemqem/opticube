import re


def format_card_name(card_name: str) -> str:
    # Convert to lower case
    card_name = card_name.lower().strip()
    # Replace single quotes with nothing
    card_name = card_name.replace("'", "")
    # Replace non-alphanumeric characters with underscores
    card_name = re.sub(r'\W+', '-', card_name)
    # Remove leading/trailing underscores
    card_name = card_name.strip('-')
    return card_name