import json
from pathlib import Path

from bs4 import BeautifulSoup


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def find_faq_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    for script in soup.find_all("script"):
        content = script.string or script.get_text()

        if '"faqData"' not in content:
            continue

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            continue

        page_props = data["props"]["pageProps"]

        return page_props["faqData"]

    return None

def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(" ", strip=True)

def save_records(records, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

def main():
    file_path = (
        RAW_DIR
        / "ather_electric_scooter_faqs___support_&_help_center.html"
    )

    faq_data = find_faq_data(file_path)

    if faq_data is None:
        print("FAQ data not found.")
        return

    print("FAQ data found.")

    records = []
    for category, groups in faq_data.items():
        for group in groups:
            for question in group["questions"]:
                if question["answer"] == None:
                    print("WARNING: Missing answer")
                    print("Category:", category)
                    print("Group:", group["name"])
                    print("FAQ ID:", question["id"])
                    print("Question:", clean_html(question["question"]))
                    continue

                record = {
                    "faq_id": question["id"],
                    "category": category,
                    "group": group["name"],
                    "question": clean_html(question["question"]),
                    "answer": clean_html(question["answer"]),
                    "source_document": file_path.stem,
                    "source_type": "official",
                }

                records.append(record)
    
    output_path = (PROCESSED_DIR / "ather_electric_scooter_faqs.jsonl")
    save_records(records,output_path)

    print(f"Processed {len(records)} FAQ records")
    print(f"Saved to {output_path}")
                    


if __name__ == "__main__":
    main()