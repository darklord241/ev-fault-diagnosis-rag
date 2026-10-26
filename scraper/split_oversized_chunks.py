import json
import os
import tempfile
from pathlib import Path
from transformers import AutoTokenizer

CHUNKED_DIR = Path(__file__).resolve().parent / "data" / "chunked"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TARGET_TOKENS = 200
OVERLAP_TOKENS = 30
MAX_TOKENS = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


def token_count(text):
    return len(tokenizer.encode(text, add_special_tokens=True))


def split_text_by_tokens(text):
    """Split text using token boundaries, preserving original text."""
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets = encoded["offset_mapping"]

    if not offsets:
        return [text] if text.strip() else []

    pieces = []
    start_token = 0

    while start_token < len(offsets):
        end_token = min(start_token + TARGET_TOKENS, len(offsets))

        # Leave room for the model's special tokens.
        while end_token > start_token:
            char_start = offsets[start_token][0]
            char_end = offsets[end_token - 1][1]
            piece = text[char_start:char_end].strip()

            if piece and token_count(piece) <= MAX_TOKENS:
                break
            end_token -= 1

        if end_token <= start_token:
            raise ValueError(
                f"Could not split text safely near token {start_token}"
            )

        pieces.append(piece)

        if end_token == len(offsets):
            break

        next_start = max(start_token + 1, end_token - OVERLAP_TOKENS)
        start_token = next_start

    return pieces


def main():
    oversized_count = 0
    new_piece_count = 0
    changed_files = 0

    for path in sorted(CHUNKED_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as file:
            records = [json.loads(line) for line in file if line.strip()]

        updated = []
        changed = False

        for record in records:
            text = record.get("text", "")

            if not isinstance(text, str) or token_count(text) <= MAX_TOKENS:
                updated.append(record)
                continue

            oversized_count += 1
            pieces = split_text_by_tokens(text)

            if len(pieces) < 2:
                raise ValueError(
                    f"{path.name}: {record.get('chunk_id')} "
                    "is oversized but could not be split"
                )

            for index, piece in enumerate(pieces):
                new_record = record.copy()
                new_record["chunk_id"] = (
                    f"{record['chunk_id']}_split_{index}"
                )
                new_record["text"] = piece
                updated.append(new_record)

                if token_count(piece) > MAX_TOKENS:
                    raise ValueError(
                        f"Still oversized: {new_record['chunk_id']}"
                    )

            new_piece_count += len(pieces)
            changed = True

            print(
                f"Split {record['chunk_id']} "
                f"({token_count(text)} tokens) into {len(pieces)} pieces"
            )

        if changed:
            fd, temp_name = tempfile.mkstemp(
                dir=path.parent, suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as file:
                    for record in updated:
                        file.write(
                            json.dumps(record, ensure_ascii=False) + "\n"
                        )
                os.replace(temp_name, path)
            except Exception:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
                raise

            changed_files += 1

    print("\nRepair complete.")
    print(f"Oversized chunks split: {oversized_count}")
    print(f"Replacement pieces created: {new_piece_count}")
    print(f"Files updated: {changed_files}")


if __name__ == "__main__":
    main()
