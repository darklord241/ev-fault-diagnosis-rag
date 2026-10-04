import subprocess
import json
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def extract_pdf_pages(pdf_path):
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        check=True
    )
    pages = result.stdout.split("\f")

    if pages and not pages[-1].strip():
        pages.pop()

    return pages

def save_pdf_pages(pages, pdf_path):
    output_path = PROCESSED_DIR / f"{pdf_path.stem}.jsonl"
    with open(output_path, "w", encoding="utf-8") as file:
        for page_number, text in enumerate(pages, start=1):
            record = {
                "record_id": f"{pdf_path.stem}_page_{page_number}",
                "source_document": pdf_path.stem,
                "source_type": "official",
                "content_type": "pdf",
                "title": pdf_path.stem,
                "text": text.strip(),
                "metadata": {
                    "category": None,
                    "group": None,
                    "page": page_number
                }
            }

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    return output_path

def main():
    pdf_files = list(RAW_DIR.glob("*.pdf"))
    for pdf_path in pdf_files:
        print("=" * 60)
        print(f"Processing: {pdf_path.name}")

        pages = extract_pdf_pages(pdf_path)
        output_path = save_pdf_pages(pages, pdf_path)

        print(f"Processed {len(pages)} PDF pages")
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()