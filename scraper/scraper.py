import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
import requests

# Configuration
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_DIR = DATA_DIR / "metadata"

RAW_DIR.mkdir(parents=True,exist_ok=True)
METADATA_DIR.mkdir(parents=True,exist_ok=True)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; EVTechnicalResearchBot/1.0; "
        "Educational research project)"
    )
}

# Official Ather sources
SOURCES = [
    {
        "url": "https://www.atherenergy.com/faq",
        "title": "Ather FAQ",
        "document_type": "faq",
        "vehicle_model": None,
    },
    {
        "url": "https://www.atherenergy.com/faq/charging-grid/charging",
        "title": "Ather Charging FAQs",
        "document_type": "faq",
        "vehicle_model": None,
    },
    {
        "url": "https://www.atherenergy.com/faq/owning-ather/ather_roadside_assistance",
        "title": "Ather Roadside Assistance FAQs",
        "document_type": "faq",
        "vehicle_model": None,
    },
    {
        "url": "https://www.atherenergy.com/faq/explore-buy/ather-450-series",
        "title": "Ather 450 Series FAQs",
        "document_type": "faq",
        "vehicle_model": "Ather 450 Series",
    },
    {
        "url": "https://www.atherenergy.com/faq/owning-ather/ather-battery-warranty",
        "title": "Ather Battery Warranty FAQs",
        "document_type": "warranty",
        "vehicle_model": None,
    },
    {
        "url": "https://www.atherenergy.com/faq/owning-ather/ecw-warranty",
        "title": "Ather Extended Comprehensive Warranty FAQs",
        "document_type": "warranty",
        "vehicle_model": None,
    },
    {
        "url": "https://media.atherenergy.com/Warranty_Policy_2024.pdf",
        "title": "Ather Warranty Policy 2024",
        "document_type": "warranty_document",
        "vehicle_model": None,
    },
    {
        "url": "https://assets.atherenergy.com/subs_terms_and_condition.pdf",
        "title": "Ather Subscription Terms and Conditions",
        "document_type": "terms_and_conditions",
        "vehicle_model": None,
    },
]

# Utility functions
def make_filename(title, extension):    
    # Convert a title into a safe filename.
    filename = title.lower()
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789-_"
    filteredChars = []
    for char in filename:
        if char in allowed or char == " ":
            filteredChars.append(char)

    filename = "".join(filteredChars)

    # gpt gave very complicated method for the above code 
    #filename = "".join(
    #   char if char in allowed or char == " " else ""
    #   for char in filename
    #)

    filename = filename.replace(" ", "_")
    return f"{filename}{extension}"


def get_extension(response):
    # Decide whether the downloaded resource is HTML, PDF, etc.
    content_type = response.headers.get("Content-Type", "").lower()
    if "application/pdf" in content_type:
        return ".pdf"
    if "text/html" in content_type:
        return ".html"
    return ".bin"

# Download function
def download_source(source):
    url = source["url"]

    print(f"\nDownloading:")
    print(url)

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30,
            allow_redirects=True,
        )

        response.raise_for_status()

    except requests.RequestException as error:
        print(f"ERROR: {error}")
        return

    extension = get_extension(response)

    filename = make_filename(
        source["title"],
        extension
    )

    raw_path = RAW_DIR / filename

    # Save the original downloaded content
    with open(raw_path, "wb") as file:
        file.write(response.content)

    # Metadata
    metadata = {
        "document_id": raw_path.stem,
        "title": source["title"],
        "source_type": "official",
        "manufacturer": "Ather Energy",
        "vehicle_model": source["vehicle_model"],
        "document_type": source["document_type"],
        "url": url,
        "final_url": response.url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "content_type": response.headers.get("Content-Type"),
        "file_path": str(raw_path),
    }

    metadata_path = METADATA_DIR / f"{raw_path.stem}.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Saved: {raw_path}")
    print(f"Metadata: {metadata_path}")


def main():
    print("Starting Ather official source collector...")
    print(f"Sources to collect: {len(SOURCES)}")

    for source in SOURCES:
        download_source(source)

        # Small delay between requests
        time.sleep(2)

    print("\nCollection complete.")


if __name__ == "__main__":
    main()