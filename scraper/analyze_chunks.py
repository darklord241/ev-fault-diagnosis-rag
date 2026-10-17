import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

for file_path in PROCESSED_DIR.glob("*.jsonl"):

    lengths = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            word_count = len(record["text"].split())
            lengths.append(word_count)

    print("=" * 60)
    print(file_path.name)
    print("Records:", len(lengths))
    print("Min:", min(lengths))
    print("Max:", max(lengths))
    print("Average:", round(sum(lengths) / len(lengths), 1))