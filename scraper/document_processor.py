import json
from pathlib import Path

from bs4 import BeautifulSoup

from provenance import get_provenance, load_sidecar_metadata


DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_DIR = DATA_DIR / "metadata"
PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def find_faq_groups(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    for script in soup.find_all("script"):
        content = script.string or script.get_text()

        if '"faqData"' not in content and '"topicData"' not in content:
            continue

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            continue

        page_props = data["props"]["pageProps"]

        if "faqData" in page_props:
            faq_data = page_props["faqData"]

            groups = []

            for category, category_groups in faq_data.items():
                for group in category_groups:
                    groups.append((category, group))

            return groups

        if "topicData" in page_props:
            topic_data = page_props["topicData"]

            groups = []

            for group in topic_data:
                category = group["category"]["slug"]

                groups.append((category, group))

            return groups

    return None

def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(" ", strip=True)

def save_records(records, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

def main():
    html_files = list(RAW_DIR.glob("*.html"))
    sidecars = load_sidecar_metadata(METADATA_DIR)

    for file_path in html_files:
        print("=" * 60)
        print(f"Processing: {file_path.name}")

        provenance = get_provenance(sidecars, file_path.stem)
        groups = find_faq_groups(file_path)

        if groups is None:
            print("FAQ data not found.")
            continue

        records = []
        skipped = 0

        for category, group in groups:
            for question in group["questions"]:

                if question["answer"] is None:
                    skipped += 1
                    print("WARNING: Missing answer")
                    print("Category:", category)
                    print("Group:", group["name"])
                    print("FAQ ID:", question["id"])
                    print("Question:", clean_html(question["question"]))
                    continue

                question_text = clean_html(question["question"])
                answer_text = clean_html(question["answer"])

                record = {
                    "record_id": f"{file_path.stem}_{question['id']}",
                    "source_document": file_path.stem,
                    "source_type": provenance["source_type"],
                    "content_type": "faq",
                    "title": question_text,
                    "text": f"{question_text}\n\n{answer_text}",
                    "provenance": provenance,
                    "metadata": {
                        "category": category,
                        "group": group["name"],
                        "page": None
                    }
                }

                records.append(record)

        output_path = PROCESSED_DIR / f"{file_path.stem}.jsonl"

        save_records(records, output_path)

        print(f"Processed {len(records)} FAQ records")
        print(f"Skipped {skipped} records with missing answers")
        print(f"Saved to {output_path}")
                    


if __name__ == "__main__":
    main()