import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from bs4 import BeautifulSoup

# Note
DATA_DIR = Path(__file__).resolve().parent / "data"
INCOMING_DIR = DATA_DIR / "incoming"
RAW_DIR = DATA_DIR / "raw"
METADATA_DIR = DATA_DIR / "metadata"

RAW_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Note
SOURCES = {
    "Ather Electric Scooter FAQs _ Support & Help Center.html": {
        "url": "https://www.atherenergy.com/faq",
        "title": "Ather Electric Scooter FAQs - Support & Help Center",
        "document_type": "faq",
        "vehicle_model": None,
    },

    "Ather Charging FAQs.html": {
        "url": "https://www.atherenergy.com/faq/charging-grid/charging",
        "title": "Ather Charging FAQs",
        "document_type": "faq",
        "vehicle_model": None,
    },

    "Ather Roadside Assistance FAQs.html": {
        "url": "https://www.atherenergy.com/faq/owning-ather/ather_roadside_assistance",
        "title": "Ather Roadside Assistance FAQs",
        "document_type": "faq",
        "vehicle_model": None,
    },

    "Ather 450 Series FAQs _ 450X vs 450S, Charging, Warranty.html": {
        "url": "https://www.atherenergy.com/faq/explore-buy/ather-450-series",
        "title": "Ather 450 Series FAQs",
        "document_type": "faq",
        "vehicle_model": "Ather 450 Series",
    },

    "Battery Warranty Details for Ather Scooters _ FAQs.html": {
        "url": "https://www.atherenergy.com/faq/owning-ather/ather-battery-warranty",
        "title": "Battery Warranty Details for Ather Scooters",
        "document_type": "warranty",
        "vehicle_model": None,
    },

    "Ather ECW Warranty FAQs.html": {
        "url": "https://www.atherenergy.com/faq/owning-ather/ecw-warranty",
        "title": "Ather Extended Comprehensive Warranty FAQs",
        "document_type": "warranty",
        "vehicle_model": None,
    },
}

# Note
def make_document_id(filename):
    return Path(filename).stem.lower().replace(" ", "_")

# Note
def process_html(filename, source_info):
    incoming_path = INCOMING_DIR / filename

    if not incoming_path.exists():
        print(f"WARNING: File not found: {filename}")
        return

    document_id = make_document_id(filename)

    # Note
    raw_filename = f"{document_id}.html"
    raw_path = RAW_DIR / raw_filename

    shutil.copy2(incoming_path, raw_path)

    # Note
    with open(incoming_path, "r", encoding="utf-8", errors="ignore") as file:
        html = file.read()

    soup = BeautifulSoup(html, "html.parser")

    html_title = None

    if soup.title:
        html_title = soup.title.get_text(strip=True)

    metadata = {
        "document_id": document_id,
        "title": source_info["title"],
        "html_title": html_title,
        "source_type": "official",
        "manufacturer": "Ather Energy",
        "vehicle_model": source_info["vehicle_model"],
        "document_type": source_info["document_type"],
        "url": source_info["url"],
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "file_path": str(raw_path),
        "original_filename": filename,
    }

    metadata_path = METADATA_DIR / f"{document_id}.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Processed: {filename}")
    print(f"  → {raw_path}")
    print(f"  → {metadata_path}")

def main():
    print("Processing collected Ather HTML pages...\n")
    for filename, source_info in SOURCES.items():
        process_html(filename, source_info)

    print("\nHTML processing complete.")

if __name__ == "__main__":
    main()