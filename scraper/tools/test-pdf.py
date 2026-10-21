from pathlib import Path

from pypdf import PdfReader

pdf_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "ather_warranty_policy_2024.pdf"
)

reader = PdfReader(pdf_path)

print("Number of pages:", len(reader.pages))

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()

    print("=" * 60)
    print(f"PAGE {page_number}")
    print("=" * 60)
    print(text[:1000])