from pathlib import Path
from tqdm import tqdm

def process_file(input_path: Path, output_path: Path) -> None:
    seen_cards = {}
    
    # Read the file and process each line
    with input_path.open('r') as file:

        lines = file.readlines()
        for line in tqdm(lines, desc="Processing lines"):
            name, count = line.strip().split(',')
            count = int(count)
            if name in seen_cards:
                assert seen_cards[name] < count, f"Existing count for {name} is not less than the new count"
            seen_cards[name] = count

    assert len(seen_cards) < 100_000, "Too many seen cards"

    # Write the processed data to the output file
    with output_path.open('w') as file:
        for name, count in tqdm(sorted(seen_cards.items(), key=lambda item: item[1], reverse=True), desc="Writing to file"):
            file.write(f'{name},{count}\n')
if __name__ == "__main__":
    input_file = Path('seen_cards.txt')
    output_file = Path('seen_cards_small.txt')
    process_file(input_file, output_file)
