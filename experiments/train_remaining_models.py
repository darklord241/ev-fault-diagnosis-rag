"""Train remaining 3 domain adaptation embedding models for comparison.

Models:
1. minilm_ft_hard_neg: Multiple Negatives Ranking with in-batch + BM25-mined hard negatives
2. minilm_ft_margin: Triplet Margin Loss with cosine distance margin
3. minilm_ft_augmented: Domain adaptation with technical pairs + colloquial paraphrases
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

os.environ["HF_TOKEN"] = ""
BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TRAIN_PATH = ROOT_DIR / "experiments" / "datasets" / "domain_adaptation_train.json"
VAL_PATH = ROOT_DIR / "experiments" / "datasets" / "domain_adaptation_val.json"
MODELS_DIR = ROOT_DIR / "experiments" / "models"


class PairDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_texts(model, tokenizer, texts: List[str], device: str, batch_size: int = 32) -> torch.Tensor:
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            out = model(**encoded)
            embs = mean_pooling(out, encoded["attention_mask"])
            embs = F.normalize(embs, p=2, dim=1)
            all_embeddings.append(embs)
    return torch.cat(all_embeddings, dim=0)


def evaluate_on_val(model, tokenizer, val_pairs: List[Dict[str, Any]], device: str) -> Dict[str, float]:
    val_queries = [p["query"] for p in val_pairs]
    val_passages = [p["passage"] for p in val_pairs]

    q_embs = encode_texts(model, tokenizer, val_queries, device)
    p_embs = encode_texts(model, tokenizer, val_passages, device)

    sim_matrix = torch.mm(q_embs, p_embs.t())
    n = len(val_pairs)
    hit_1 = 0
    hit_3 = 0
    hit_5 = 0
    mrr = 0.0

    for i in range(n):
        scores = sim_matrix[i].cpu().tolist()
        ranked = sorted(range(n), key=lambda idx: scores[idx], reverse=True)
        if i in ranked[:1]:
            hit_1 += 1
        if i in ranked[:3]:
            hit_3 += 1
        if i in ranked[:5]:
            hit_5 += 1
            rank = ranked[:5].index(i) + 1
            mrr += 1.0 / rank

    return {
        "val_hit_1": hit_1 / n,
        "val_hit_3": hit_3 / n,
        "val_hit_5": hit_5 / n,
        "val_mrr_5": mrr / n,
    }


def mine_hard_negatives(pairs: List[Dict[str, Any]]) -> List[str]:
    def tokenize(text):
        return set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))

    passages = [p["passage"] for p in pairs]
    passage_tokens = [tokenize(p) for p in passages]

    hard_negatives = []
    for i, item in enumerate(pairs):
        q_tokens = tokenize(item["query"])
        best_score = -1
        best_j = -1
        for j, p_tok in enumerate(passage_tokens):
            if i == j:
                continue
            overlap = len(q_tokens & p_tok)
            if overlap > best_score:
                best_score = overlap
                best_j = j
        hard_negatives.append(passages[best_j if best_j >= 0 else (i + 1) % len(passages)])
    return hard_negatives


def create_paraphrased_data(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Rule-based colloquial reformulation patterns for technical queries
    augmented = list(pairs)
    for p in pairs:
        q = p["query"]
        para = q
        # Common reformulations
        para = re.sub(r"^What instructions are provided for (.*?)\?$", r"how to do \1", para, flags=re.IGNORECASE)
        para = re.sub(r"^What are the operational guidelines and speed limits for (.*?)\?$", r"max speed and rules for \1", para, flags=re.IGNORECASE)
        para = re.sub(r"^What steps should be followed if (.*?)\?$", r"what to do if \1", para, flags=re.IGNORECASE)
        para = re.sub(r"^What precautions should be taken when (.*?)\?$", r"precautions while \1", para, flags=re.IGNORECASE)
        para = re.sub(r"^Where is the (.*?) located and how are (.*?) replaced\?$", r"where is \1 and changing \2", para, flags=re.IGNORECASE)
        para = re.sub(r"^How does (.*?) regulate vehicle speed\?$", r"\1 speed control", para, flags=re.IGNORECASE)
        para = re.sub(r"^What tyre pressure is recommended for (.*?)\?$", r"proper air pressure for tyres", para, flags=re.IGNORECASE)
        para = re.sub(r"^How does regenerative braking help recover (.*?)\?$", r"regen braking recharging battery", para, flags=re.IGNORECASE)
        para = re.sub(r"^What does the manual state regarding (.*?)\?$", r"\1 manual guide", para, flags=re.IGNORECASE)
        para = re.sub(r"^Can I (.*?)\?$", r"is it possible to \1", para, flags=re.IGNORECASE)
        para = re.sub(r"^Will I get (.*?)\?$", r"do i get \1", para, flags=re.IGNORECASE)
        
        if para != q:
            augmented.append({
                "query": para.strip(" ?"),
                "passage": p["passage"],
                "source_document": p["source_document"],
                "chunk_id": p["chunk_id"],
            })
    return augmented


def save_checkpoint(model, tokenizer, output_dir: Path, config: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved checkpoint to {output_dir}")


def train_model(
    model_name: str,
    method_name: str,
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    device: str,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
) -> None:
    print("\n" + "=" * 70)
    print(f"  TRAINING: {model_name} (Method: {method_name})")
    print(f"  Train samples: {len(train_data)} | Val samples: {len(val_data)}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=False)
    model = AutoModel.from_pretrained(BASE_MODEL_NAME, token=False)
    model.to(device)

    output_dir = MODELS_DIR / model_name
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    val_init = evaluate_on_val(model, tokenizer, val_data, device)
    print(f"  Pre-train Val MRR@5: {val_init['val_mrr_5']:.4f} | Hit@1: {val_init['val_hit_1']:.4f}")

    best_mrr = val_init["val_mrr_5"]

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        random.shuffle(train_data)
        num_batches = len(train_data) // batch_size

        for b in range(num_batches):
            batch = train_data[b * batch_size : (b + 1) * batch_size]
            queries = [item["query"] for item in batch]
            positives = [item["passage"] for item in batch]

            q_enc = tokenizer(queries, padding=True, truncation=True, max_length=256, return_tensors="pt")
            p_enc = tokenizer(positives, padding=True, truncation=True, max_length=256, return_tensors="pt")
            q_enc = {k: v.to(device) for k, v in q_enc.items()}
            p_enc = {k: v.to(device) for k, v in p_enc.items()}

            q_emb = F.normalize(mean_pooling(model(**q_enc), q_enc["attention_mask"]), p=2, dim=1)
            p_emb = F.normalize(mean_pooling(model(**p_enc), p_enc["attention_mask"]), p=2, dim=1)

            if method_name == "hard_neg":
                negatives = [item["hard_negative"] for item in batch]
                n_enc = tokenizer(negatives, padding=True, truncation=True, max_length=256, return_tensors="pt")
                n_enc = {k: v.to(device) for k, v in n_enc.items()}
                n_emb = F.normalize(mean_pooling(model(**n_enc), n_enc["attention_mask"]), p=2, dim=1)

                sim_pos = torch.mm(q_emb, p_emb.t()) * 20.0
                sim_neg = (q_emb * n_emb).sum(dim=1, keepdim=True) * 20.0
                logits = torch.cat([sim_pos, sim_neg], dim=1)
                labels = torch.arange(len(batch), device=device)
                loss = F.cross_entropy(logits, labels)

            elif method_name == "margin":
                negatives = [item["hard_negative"] for item in batch]
                n_enc = tokenizer(negatives, padding=True, truncation=True, max_length=256, return_tensors="pt")
                n_enc = {k: v.to(device) for k, v in n_enc.items()}
                n_emb = F.normalize(mean_pooling(model(**n_enc), n_enc["attention_mask"]), p=2, dim=1)

                pos_sim = (q_emb * p_emb).sum(dim=1)
                neg_sim = (q_emb * n_emb).sum(dim=1)
                loss = F.relu(neg_sim - pos_sim + 0.3).mean()

            else:  # augmented / inbatch
                sim_matrix = torch.mm(q_emb, p_emb.t()) * 20.0
                labels = torch.arange(len(batch), device=device)
                loss = F.cross_entropy(sim_matrix, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(num_batches, 1)
        val_metrics = evaluate_on_val(model, tokenizer, val_data, device)
        cur_mrr = val_metrics["val_mrr_5"]
        print(f"  Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} | Val MRR@5: {cur_mrr:.4f} | Hit@1: {val_metrics['val_hit_1']:.4f}")

        if cur_mrr >= best_mrr or epoch == epochs:
            best_mrr = cur_mrr
            save_checkpoint(
                model,
                tokenizer,
                output_dir,
                {
                    "model_name": model_name,
                    "method": method_name,
                    "epoch": epoch,
                    "val_mrr_5": cur_mrr,
                    "val_hit_1": val_metrics["val_hit_1"],
                    "lr": lr,
                    "batch_size": batch_size,
                },
            )


def main() -> None:
    random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_pairs = json.load(f)["pairs"]
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        val_pairs = json.load(f)["pairs"]

    # 1. Mine hard negatives for training pairs
    print("\nMining hard negatives for training pairs...")
    hard_negs = mine_hard_negatives(train_pairs)
    triplet_train_data = []
    for item, hn in zip(train_pairs, hard_negs):
        triplet_train_data.append({**item, "hard_negative": hn})
    print(f"Prepared {len(triplet_train_data)} triplets with hard negatives.")

    # 2. Prepare paraphrase-augmented dataset
    print("\nPreparing paraphrase-augmented dataset...")
    augmented_train_data = create_paraphrased_data(train_pairs)
    print(f"Prepared {len(augmented_train_data)} augmented training pairs.")

    # Model 2: minilm_ft_hard_neg
    train_model(
        model_name="minilm_ft_hard_neg",
        method_name="hard_neg",
        train_data=triplet_train_data,
        val_data=val_pairs,
        device=device,
        epochs=5,
        batch_size=16,
        lr=2e-5,
    )

    # Model 3: minilm_ft_margin
    train_model(
        model_name="minilm_ft_margin",
        method_name="margin",
        train_data=triplet_train_data,
        val_data=val_pairs,
        device=device,
        epochs=5,
        batch_size=16,
        lr=2e-5,
    )

    # Model 4: minilm_ft_augmented
    train_model(
        model_name="minilm_ft_augmented",
        method_name="augmented",
        train_data=augmented_train_data,
        val_data=val_pairs,
        device=device,
        epochs=5,
        batch_size=16,
        lr=2e-5,
    )

    print("\n" + "=" * 70)
    print("ALL 3 MODELS SUCCESSFULLY TRAINED AND SAVED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
