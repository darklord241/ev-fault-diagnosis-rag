import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


SCRAPER_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = (
    SCRAPER_DIR / "data" / "incoming" / "ather-batch" / "extracted"
)
SPECIFICATION_REVIEW_TARGETS = {
    "Ather-450-Spec-Sheet-Oct-2024.pdf",
    "Ather-450-specsheet.pdf",
    "Rizta_specifications.pdf",
}


def parse_pdfinfo(output):
    metadata = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def page_text_counts(output, expected_page_count=None):
    page_texts = output.split("\f")
    if page_texts and not page_texts[-1].strip():
        page_texts.pop()

    page_count = (
        expected_page_count
        if expected_page_count is not None
        else len(page_texts)
    )
    pages = []
    for page_number in range(1, page_count + 1):
        if page_number > len(page_texts):
            pages.append(
                {
                    "page_number": page_number,
                    "non_whitespace_chars": None,
                    "status": "missing_from_extraction",
                    "human_review_required": True,
                }
            )
            continue

        char_count = sum(
            not character.isspace() for character in page_texts[page_number - 1]
        )
        pages.append(
            {
                "page_number": page_number,
                "non_whitespace_chars": char_count,
                "status": (
                    "text_extracted"
                    if char_count
                    else "zero_text_human_review"
                ),
                "human_review_required": char_count == 0,
            }
        )

    return pages, len(page_texts)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_tool(command):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as error:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": str(error),
            "error": str(error),
        }

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": None,
    }


def audit_pdf(path):
    report = {
        "filename": path.name,
        "sha256": None,
        "pdf_metadata": {},
        "page_count": None,
        "extraction_page_count": None,
        "pages": [],
        "specification_table_review_target": (
            path.name in SPECIFICATION_REVIEW_TARGETS
        ),
        "errors": [],
    }

    try:
        report["sha256"] = sha256_file(path)
    except OSError as error:
        report["errors"].append(
            {"stage": "sha256", "message": str(error)}
        )

    info = run_tool(["pdfinfo", str(path)])
    report["pdf_metadata"] = parse_pdfinfo(info["stdout"])
    if info["returncode"] == 0:
        raw_page_count = report["pdf_metadata"].get("Pages")
        try:
            report["page_count"] = int(raw_page_count)
        except (TypeError, ValueError):
            report["errors"].append(
                {
                    "stage": "pdfinfo",
                    "message": "pdfinfo did not provide a valid Pages value",
                }
            )
    else:
        report["errors"].append(
            {
                "stage": "pdfinfo",
                "returncode": info["returncode"],
                "message": info["error"] or info["stderr"].strip(),
            }
        )

    extraction = run_tool(["pdftotext", "-layout", str(path), "-"])
    pages, extraction_page_count = page_text_counts(
        extraction["stdout"], report["page_count"]
    )
    report["extraction_page_count"] = extraction_page_count
    if extraction["returncode"] != 0:
        report["errors"].append(
            {
                "stage": "pdftotext",
                "returncode": extraction["returncode"],
                "message": extraction["error"] or extraction["stderr"].strip(),
            }
        )
        if not pages:
            pages = [
                {
                    "page_number": None,
                    "non_whitespace_chars": None,
                    "status": "extraction_failed_page_count_unknown",
                    "human_review_required": True,
                }
            ]
        else:
            for page in pages:
                page["status"] = "extraction_failed"
                page["human_review_required"] = True
                page["non_whitespace_chars"] = None
    report["pages"] = pages

    if (
        report["page_count"] is not None
        and extraction_page_count != report["page_count"]
    ):
        report["errors"].append(
            {
                "stage": "page_count",
                "message": (
                    f"pdfinfo reports {report['page_count']} pages but "
                    f"pdftotext output split into {extraction_page_count}"
                ),
            }
        )

    return report


def build_report(input_dir):
    documents = []
    seen_hashes = {}

    if not input_dir.is_dir():
        return {
            "input_directory": str(input_dir),
            "pdf_count": 0,
            "unique_sha256_count": 0,
            "count_semantics": {
                "pdf_count": "number of top-level PDF files audited, including duplicate content",
                "unique_sha256_count": "number of distinct SHA-256 hashes successfully calculated",
            },
            "documents": [],
            "errors": [
                {
                    "stage": "input_directory",
                    "message": f"Directory does not exist: {input_dir}",
                }
            ],
        }

    for path in sorted(input_dir.glob("*.pdf")):
        report = audit_pdf(path)
        digest = report["sha256"]
        report["duplicate_of"] = (
            seen_hashes.get(digest) if digest is not None else None
        )
        if digest is not None and digest not in seen_hashes:
            seen_hashes[digest] = path.name
        documents.append(report)

    return {
        "input_directory": str(input_dir),
        "pdf_count": len(documents),
        "unique_sha256_count": len(seen_hashes),
        "count_semantics": {
            "pdf_count": "number of top-level PDF files audited, including duplicate content",
            "unique_sha256_count": "number of distinct SHA-256 hashes successfully calculated",
        },
        "documents": documents,
        "errors": [],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Audit PDF metadata and pdftotext page output without "
            "modifying the PDFs or pipeline data."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing top-level PDF files to audit.",
    )
    args = parser.parse_args(argv)

    report = build_report(args.input_dir)
    report["tools"] = {
        "pdfinfo": shutil.which("pdfinfo"),
        "pdftotext": shutil.which("pdftotext"),
    }
    json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 1 if report["errors"] or any(
        document["errors"] for document in report["documents"]
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
