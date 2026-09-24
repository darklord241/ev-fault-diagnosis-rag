import re
from pathlib import Path
from bs4 import BeautifulSoup


RAW_DIR = Path("data/raw")

KEYWORDS = [
    '"questions"',
    '"question"',
    '"answer"',
    '"faq"',
    '"faqs"',
]


for file_path in RAW_DIR.glob("*.html"):
    print("=" * 70)
    print(file_path.name)

    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    print("Title:", soup.title.get_text(strip=True) if soup.title else "None")

    scripts = soup.find_all("script")
    print("Scripts:", len(scripts))

    found_keywords = set()

    for script in scripts:
        content = script.string or script.get_text()

        for keyword in KEYWORDS:
            if keyword in content:
                found_keywords.add(keyword)

    if found_keywords:
        print("Keywords found:", ", ".join(found_keywords))
    else:
        print("No target keywords found")

    for script_number, script in enumerate(scripts):
        content = script.string or script.get_text()

        for keyword in KEYWORDS:
            position = content.find(keyword)

            if position != -1:
                start = max(0, position - 150)
                end = min(len(content), position + 350)

                print()
                print(f"First '{keyword}' found in script {script_number}:")
                print(content[start:end])
                break

        else:
            continue

        break