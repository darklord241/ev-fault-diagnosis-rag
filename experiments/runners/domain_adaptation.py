"""Train MiniLM with in-batch negatives on a grouped EV evidence split."""

import argparse
import csv
import hashlib
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

from experiments.evaluation import load_evaluation_inputs, normalize_text, ranking_metrics
from scraper.vector_search.retrieval import load_chunks

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
METRICS = ("hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5", "seconds")


def evidence_groups(records, cases, annotations):
    parents = {row["chunk_id"]: row["chunk_id"] for row in records}

    def root(key):
        while key != parents[key]:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def merge(a, b):
        parents[root(a)] = root(b)

    seen = {}
    for row in records:
        key = row["chunk_id"]
        signatures = (
            ("record", row["source_document"], row["record_id"]),
            ("text", normalize_text(row["text"])),
        )
        for signature in signatures:
            merge(key, seen.setdefault(signature, key))
    for case in cases:
        first, *other = case["gold_chunk_ids"]
        for key in other:
            merge(first, key)
        for fact in annotations[case["id"]]["key_facts"]:
            signature = ("fact", normalize_text(fact))
            merge(first, seen.setdefault(signature, first))

    groups = defaultdict(lambda: {"chunks": [], "cases": []})
    for key in parents:
        groups[root(key)]["chunks"].append(key)
    for case in cases:
        groups[root(case["gold_chunk_ids"][0])]["cases"].append(case)
    return list(groups.values())


def partition(groups, annotations, fraction, seed):
    """Select intact groups close to the requested passage and query proportions."""
    groups = list(groups)
    random.Random(seed).shuffle(groups)
    query_target = round(sum(len(g["cases"]) for g in groups) * fraction)
    chunk_target = round(sum(len(g["chunks"]) for g in groups) * fraction)
    exact_target = round(sum(
        annotations[c["id"]]["query_type"] == "exact_term"
        for g in groups for c in g["cases"]
    ) * fraction)
    options = {(0, 0): ()}
    for index, group in enumerate(groups):
        n = len(group["cases"])
        if not n:
            continue
        exact = sum(annotations[c["id"]]["query_type"] == "exact_term" for c in group["cases"])
        for (previous_n, previous_exact), indices in tuple(options.items()):
            options.setdefault((previous_n + n, previous_exact + exact), indices + (index,))
    best = min(options, key=lambda item: (
        abs(item[0] - query_target), abs(item[1] - exact_target)
    ))
    selected = set(options[best])
    budget = max(0, chunk_target - sum(len(groups[i]["chunks"]) for i in selected))
    subsets = {0: None}
    for index, group in enumerate(groups):
        if group["cases"]:
            continue
        size = len(group["chunks"])
        for previous in tuple(subsets):
            if previous + size <= budget:
                subsets.setdefault(previous + size, (previous, index))
        if budget in subsets:
            break
    total = max(subsets)
    while total:
        total, index = subsets[total]
        selected.add(index)
    return (
        [group for i, group in enumerate(groups) if i not in selected],
        [group for i, group in enumerate(groups) if i in selected],
    )


def evaluate(model, groups, text_by_id, annotations, label):
    ids = [key for group in groups for key in group["chunks"]]
    cases = [case for group in groups for case in group["cases"]]
    model.eval()
    vectors = model.encode([text_by_id[key] for key in ids], normalize_embeddings=True,
                           batch_size=32, convert_to_tensor=True, show_progress_bar=False)
    rows = []
    for case in cases:
        start = time.perf_counter()
        query_vector = model.encode(case["query"], normalize_embeddings=True, convert_to_tensor=True)
        scores = vectors @ query_vector
        ranked = [ids[i] for i in torch.argsort(scores, descending=True, stable=True)[:5].tolist()]
        elapsed = time.perf_counter() - start
        rows.append({
            "model": label,
            "query_id": case["id"],
            "query_type": annotations[case["id"]]["query_type"],
            **ranking_metrics(ranked, gold_ids=set(case["gold_chunk_ids"])),
            "seconds": elapsed,
            "top5_ids": ranked,
        })
    return rows


def train_epoch(model, loss, optimizer, groups, text_by_id, seed, batch_size):
    rng = random.Random(seed)
    queues = []
    for group in groups:
        cases = group["cases"][:]
        if cases:
            rng.shuffle(cases)
            queues.append(cases)
    values = []
    used = 0
    model.train()
    while len(queues) >= 2:
        rng.shuffle(queues)
        queues.sort(key=len, reverse=True)
        cases = [queue.pop() for queue in queues[:batch_size]]
        queues = [queue for queue in queues if queue]
        questions = [case["query"] for case in cases]
        passages = [text_by_id[rng.choice(case["gold_chunk_ids"])] for case in cases]
        features = [
            {key: value.to(model.device) if torch.is_tensor(value) else value
             for key, value in model.preprocess(texts).items()}
            for texts in (questions, passages)
        ]
        optimizer.zero_grad(set_to_none=True)
        value = loss(features, None)
        value.backward()
        optimizer.step()
        values.append(value.item())
        used += len(cases)
    return statistics.mean(values), used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    output = Path("results/domain_adaptation") / args.run_id
    checkpoint = Path("experiments/models") / f"minilm_{args.run_id}"
    if output.exists() or checkpoint.exists():
        parser.error("The run or checkpoint already exists. Choose another run ID.")

    corpus = Path("data/chunked/benchmark_200")
    datasets = Path("experiments/datasets")
    records = load_chunks(corpus)
    cases_path = datasets / "retrieval_eval_final_200_updated.json"
    annotations_path = datasets / "evaluation_annotations_final_200_updated.json"
    cases, annotations = load_evaluation_inputs(cases_path, annotations_path, records)
    text = {row["chunk_id"]: row["text"] for row in records}
    seed, epochs, batch_size, lr = 42, 3, 16, 2e-5
    torch.manual_seed(seed)
    train_side, test = partition(evidence_groups(records, cases, annotations), annotations, .30, seed)
    train, validation = partition(train_side, annotations, .10, seed + 1)
    splits = {"training": train, "validation": validation, "test": test}
    manifest = {
        name: [{"chunks": group["chunks"], "queries": [c["id"] for c in group["cases"]]}
               for group in groups]
        for name, groups in splits.items()
    }
    for name, groups in manifest.items():
        print(name, sum(len(g["chunks"]) for g in groups), "passages",
              sum(len(g["queries"]) for g in groups), "queries", flush=True)
    assert all(any(g["queries"] for g in groups) for groups in manifest.values())
    output.mkdir(parents=True)
    (output / "split.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    config = {
        "base_model": MODEL, "seed": seed, "test_fraction": .30,
        "validation_fraction_of_training": .10, "epochs": epochs,
        "batch_size": batch_size, "learning_rate": lr,
        "loss": "MultipleNegativesRankingLoss",
        "validation_metric": "MRR@5", "test_index": "held-out passages only",
        "checkpoint": checkpoint.as_posix(), "corpus": corpus.as_posix(),
        "inputs_sha256": {
            path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [cases_path, annotations_path, *sorted(corpus.glob("*.jsonl"))]
        },
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model = SentenceTransformer(MODEL, local_files_only=True, device="cpu")
    model.max_seq_length = 256
    for passage in list(text.values()) + [case["query"] for case in cases]:
        if len(model.tokenizer(passage, truncation=False)["input_ids"]) > 256:
            raise ValueError("An input exceeds the checked 256-token limit.")

    baseline = evaluate(model, test, text, annotations, "pretrained")
    loss = MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best = -1
    history = []
    for epoch in range(1, epochs + 1):
        value, used = train_epoch(model, loss, optimizer, train, text, seed + epoch, batch_size)
        validation_rows = evaluate(model, validation, text, annotations, "validation")
        mrr = statistics.mean(row["mrr"] for row in validation_rows)
        history.append({"epoch": epoch, "loss": value, "training_queries_used": used,
                        "validation_mrr_at_5": mrr})
        print(history[-1], flush=True)
        if mrr > best:
            best = mrr
            model.save_pretrained(str(checkpoint))
    selected = SentenceTransformer(str(checkpoint), local_files_only=True, device="cpu")
    rows = baseline + evaluate(selected, test, text, annotations, "fine_tuned")
    with (output / "per_query_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({**row, "top5_ids": json.dumps(row["top5_ids"])} for row in rows)
    summary = []
    for label in ("pretrained", "fine_tuned"):
        for kind in ("overall", "exact_term", "paraphrased"):
            subset = [row for row in rows if row["model"] == label and (
                kind == "overall" or row["query_type"] == kind
            )]
            summary.append({
                "model": label, "query_type": kind, "n": len(subset),
                **{key: statistics.mean(row[key] for row in subset) for key in METRICS},
            })
    with (output / "aggregate_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
