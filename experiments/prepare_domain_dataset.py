"""Generate leak-free training and validation datasets for domain-specific embedding adaptation."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Set

from scraper.vector_search.retrieval import CHUNKED_DIR, load_chunks

SEED = 42
TARGET_TRAIN_COUNT = 300
TARGET_VAL_COUNT = 75
TARGET_TOTAL = TARGET_TRAIN_COUNT + TARGET_VAL_COUNT  # 375 total pairs

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT_DIR / "experiments" / "datasets"
BENCHMARK_PATH = DATASETS_DIR / "retrieval_eval_40.json"
TRAIN_OUTPUT_PATH = DATASETS_DIR / "domain_adaptation_train.json"
VAL_OUTPUT_PATH = DATASETS_DIR / "domain_adaptation_val.json"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_subject_from_pdf_text(text: str) -> str | None:
    """Extract a salient subject or question from a PDF manual chunk."""
    clean = clean_text(text)
    if len(clean) < 60:
        return None
    
    # Check for prominent all-caps or title headers at the beginning
    # Examples: 'UNPLUGGING IT', 'SWITCHGEAR: LEFT-HAND SIDE', 'MAINTAINING SPEED'
    match = re.match(r"^([A-Z0-9\s\-\:\,\/\(\)\.\']{4,60})(?:\s+[a-z0-9]|\n|$)", text.strip())
    if match:
        header = clean_text(match.group(1))
        # Filter out noisy single-word or index headers
        if len(header) >= 6 and header.lower() not in {"index", "quick start guide", "caution", "warning", "note"}:
            return f"What instructions are provided for {header.lower()}?"

    # Check for section topic keywords
    topics = [
        (r"park assist|reverse speed", "What are the operational guidelines and speed limits for Park Assist?"),
        (r"connector (?:feels )?stuck|pull out.*connector", "What steps should be followed if the charging connector feels stuck?"),
        (r"wet weather|braking in wet", "What precautions should be taken when braking in wet weather?"),
        (r"joystick|feature settings", "How is the joystick used to navigate scooter features and settings?"),
        (r"kill switch|unintended acceleration", "When should the motor kill switch be used during stops?"),
        (r"left-hand side|switchgear|turning indicator", "What functions are controlled by the left-hand switchgear?"),
        (r"right-hand side|switchgear|start switch", "What functions are controlled by the right-hand switchgear?"),
        (r"magic twist|accelerator grip", "How does the Magic Twist accelerator grip regulate vehicle speed?"),
        (r"pressure jet wash|storage box", "Can a pressure jet wash be used on the under-seat storage box?"),
        (r"tyre pressure|tire pressure|psi", "What tyre pressure is recommended for the electric scooter?"),
        (r"combined braking|cbs|disc brake", "How does the Combined Braking System operate on the wheels?"),
        (r"storage.*scooter|idle storage|long term", "What guidelines should be followed for long-term scooter storage?"),
        (r"washing|cleaning the scooter", "What are the recommended cleaning procedures for the vehicle exterior?"),
        (r"fuse box|blown fuse|replace.*fuse", "Where is the fuse box located and how are fuses replaced?"),
        (r"headlight|high beam|low beam", "How do I operate the high beam and low beam headlight switches?"),
        (r"dashboard reset|touchscreen reboot", "What is the procedure to reboot the touchscreen dashboard?"),
        (r"drive belt|belt tension", "What maintenance is required for the final drive belt?"),
        (r"unplugging|disconnect.*charger", "What is the correct procedure for unplugging the charging cable?"),
        (r"portable charger|charging safety", "What safety instructions apply when connecting the portable charger?"),
        (r"pillion footrest|footpeg", "How do the integrated pillion footrests operate?"),
        (r"fast charging|grid charging", "What guidelines apply when charging at fast public charging stations?"),
        (r"side stand|side stand sensor", "How does the side stand sensor prevent accidental movement?"),
        (r"battery shutdown|scooter shutdown", "How is the scooter put into deep sleep or shutdown mode?"),
        (r"first aid kit|document storage", "Where is the first aid kit stored on the scooter?"),
        (r"mirror adjustment|rear view", "How should the rear view mirrors be adjusted?"),
        (r"bluetooth pairing|sync phone", "What is the procedure for pairing a smartphone via Bluetooth?"),
        (r"regenerative braking|regen", "How does regenerative braking help recover battery charge?"),
        (r"suspension|monoshock", "What suspension setup is equipped on the front and rear wheels?"),
    ]

    lower = clean.lower()
    for pattern, query in topics:
        if re.search(pattern, lower):
            return query

    # Generic informative chunk extraction based on first sentence
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    if sentences:
        first = sentences[0]
        if 25 <= len(first) <= 120 and not first.isupper():
            first_clean = first.rstrip(".:!?")
            return f"What does the manual state regarding {first_clean.lower()}?"

    return None


def main() -> None:
    random.seed(SEED)

    # 1. Load benchmark to establish strict blacklist
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    benchmark_queries = [c["query"].strip().lower() for c in benchmark_data["cases"]]
    gold_chunk_ids: Set[str] = set()
    for c in benchmark_data["cases"]:
        for gid in (c.get("gold_chunk_ids") or []):
            gold_chunk_ids.add(gid)

    print("=" * 80)
    print("  DOMAIN ADAPTATION DATASET GENERATOR & LEAKAGE VERIFIER")
    print("=" * 80)
    print(f"Benchmark Cases: {len(benchmark_data['cases'])}")
    print(f"Gold Chunk IDs Blacklisted: {len(gold_chunk_ids)}")

    # 2. Load all 891 chunks
    chunks = load_chunks(CHUNKED_DIR)
    print(f"Total Corpus Chunks: {len(chunks)}")

    # Filter out all gold chunks
    eligible_chunks = [c for c in chunks if c["chunk_id"] not in gold_chunk_ids]
    print(f"Eligible Non-Gold Chunks: {len(eligible_chunks)}")

    extracted_pairs: List[Dict[str, Any]] = []
    seen_queries: Set[str] = set()
    seen_passages: Set[str] = set()

    # 3. Process FAQ chunks
    faq_count = 0
    for c in eligible_chunks:
        if c.get("content_type") == "faq":
            q = clean_text(c.get("title") or "")
            body = clean_text(c.get("text") or "")
            if len(q) >= 15 and len(body) >= 50:
                q_lower = q.lower()
                # Strict leakage check against benchmark queries
                if q_lower not in benchmark_queries and q_lower not in seen_queries:
                    # Ensure no heavy substring/word overlap with benchmark queries
                    words_q = set(q_lower.split())
                    overlap_flag = False
                    for bq in benchmark_queries:
                        words_bq = set(bq.split())
                        jaccard = len(words_q & words_bq) / max(len(words_q | words_bq), 1)
                        if jaccard > 0.60:
                            overlap_flag = True
                            break

                    if not overlap_flag:
                        seen_queries.add(q_lower)
                        seen_passages.add(c["chunk_id"])
                        faq_count += 1
                        extracted_pairs.append({
                            "pair_id": f"pair_faq_{c['chunk_id']}",
                            "query": q,
                            "chunk_id": c["chunk_id"],
                            "passage": body,
                            "source_document": c.get("source_document") or "Ather FAQ",
                            "source_type": "faq",
                        })

    print(f"Extracted Leak-Free FAQ Pairs: {faq_count}")

    # 4. Process PDF chunks
    pdf_count = 0
    for c in eligible_chunks:
        if c.get("content_type") == "pdf":
            body = clean_text(c.get("text") or "")
            if len(body) >= 60:
                q = extract_subject_from_pdf_text(c.get("text") or "")
                if q:
                    q_lower = q.lower()
                    if q_lower not in benchmark_queries and q_lower not in seen_queries:
                        words_q = set(q_lower.split())
                        overlap_flag = False
                        for bq in benchmark_queries:
                            words_bq = set(bq.split())
                            jaccard = len(words_q & words_bq) / max(len(words_q | words_bq), 1)
                            if jaccard > 0.60:
                                overlap_flag = True
                                break

                        if not overlap_flag:
                            seen_queries.add(q_lower)
                            seen_passages.add(c["chunk_id"])
                            pdf_count += 1
                            extracted_pairs.append({
                                "pair_id": f"pair_pdf_{c['chunk_id']}",
                                "query": q,
                                "chunk_id": c["chunk_id"],
                                "passage": body,
                                "source_document": c.get("source_document") or "Ather Manual",
                                "source_type": "pdf",
                            })

    print(f"Extracted Leak-Free PDF Pairs: {pdf_count}")
    print(f"Total Extracted Leak-Free Candidates: {len(extracted_pairs)}")

    # Deterministic shuffle
    random.shuffle(extracted_pairs)

    if len(extracted_pairs) < TARGET_TOTAL:
        raise RuntimeError(f"Insufficient pairs extracted: got {len(extracted_pairs)}, needed {TARGET_TOTAL}")

    selected_pairs = extracted_pairs[:TARGET_TOTAL]
    train_pairs = selected_pairs[:TARGET_TRAIN_COUNT]
    val_pairs = selected_pairs[TARGET_TRAIN_COUNT:TARGET_TOTAL]

    # Verify zero leakage programmatically
    for p in selected_pairs:
        assert p["chunk_id"] not in gold_chunk_ids, f"LEAKAGE DETECTED: {p['chunk_id']} in gold set!"
        assert p["query"].lower() not in benchmark_queries, f"LEAKAGE DETECTED: {p['query']} in benchmark!"

    # Save Train Dataset
    train_data = {
        "version": 1,
        "split": "train",
        "description": "Domain-specific EV technical training pairs (leak-free from 40-case benchmark)",
        "total_pairs": len(train_pairs),
        "unique_passages": len(set(p["chunk_id"] for p in train_pairs)),
        "source_documents": sorted(list(set(p["source_document"] for p in train_pairs))),
        "pairs": train_pairs,
    }
    with open(TRAIN_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)

    # Save Val Dataset
    val_data = {
        "version": 1,
        "split": "val",
        "description": "Domain-specific EV technical validation dev set (leak-free from 40-case benchmark)",
        "total_pairs": len(val_pairs),
        "unique_passages": len(set(p["chunk_id"] for p in val_pairs)),
        "source_documents": sorted(list(set(p["source_document"] for p in val_pairs))),
        "pairs": val_pairs,
    }
    with open(VAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)

    print("\n[SUCCESS] Datasets generated and verified:")
    print(f"  Training pairs:   {len(train_pairs)} (saved to {TRAIN_OUTPUT_PATH.name})")
    print(f"  Validation pairs: {len(val_pairs)} (saved to {VAL_OUTPUT_PATH.name})")
    print(f"  Unique passages:  {len(seen_passages)}")
    print(f"  Source documents: {len(train_data['source_documents'])}")
    print("  Leakage Check:    PASSED (0 gold chunk IDs, 0 benchmark queries present)")


if __name__ == "__main__":
    main()
