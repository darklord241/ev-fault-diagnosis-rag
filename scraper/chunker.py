import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKED_DIR = DATA_DIR / "chunked"
CHUNKED_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

def load_records(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    return records

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap

    return chunks

def create_chunks(record):
    # Note
    if record["content_type"] == "faq":
        return [{
            "chunk_id": f"{record['record_id']}_chunk_0",
            "record_id": record["record_id"],
            "source_document": record["source_document"],
            "source_type": record["source_type"],
            "content_type": record["content_type"],
            "title": record["title"],
            "text": record["text"],
            "provenance": record["provenance"],
            "metadata": record["metadata"]
        }]

    # Note
    if record["content_type"] == "pdf":
        text_chunks = split_text(record["text"])
        chunks = []
        for index, text in enumerate(text_chunks):
            chunks.append({
                "chunk_id": f"{record['record_id']}_chunk_{index}",
                "record_id": record["record_id"],
                "source_document": record["source_document"],
                "source_type": record["source_type"],
                "content_type": record["content_type"],
                "title": record["title"],
                "text": text,
                "provenance": record["provenance"],
                "metadata": record["metadata"]
            })

        return chunks
    
    return []

def save_chunks(chunks, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

def main():
    processed_files = list(PROCESSED_DIR.glob("*.jsonl"))
    total_chunks = 0
    for file_path in processed_files:
        print("=" * 60)
        print(f"Processing: {file_path.name}")
        records = load_records(file_path)
        chunks = []
        for record in records:
            chunks.extend(create_chunks(record))

        output_path = CHUNKED_DIR / file_path.name
        save_chunks(chunks, output_path)
        total_chunks += len(chunks)

        print(f"Records: {len(records)}")
        print(f"Chunks:  {len(chunks)}")
        print(f"Saved:   {output_path}")

    print("=" * 60)
    print(f"Total chunks: {total_chunks}")

if __name__ == "__main__":
    main()