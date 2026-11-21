"""Train domain-adapted MiniLM model using Multiple Negatives Ranking Loss (MNRL / InfoNCE).

Deterministic, self-contained PyTorch training script with validation evaluation.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

# Prevent unauthenticated token issues
os.environ["HF_TOKEN"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT_DIR / "experiments" / "datasets"
TRAIN_PATH = DATASETS_DIR / "domain_adaptation_train.json"
VAL_PATH = DATASETS_DIR / "domain_adaptation_val.json"
OUTPUT_MODEL_DIR = ROOT_DIR / "experiments" / "models" / "minilm_ev_domain_adapted"
OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Training Hyperparameters
BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MNRL_SCALE = 20.0
MAX_SEQ_LENGTH = 256


class PairDataset(Dataset):
    def __init__(self, pairs: list[dict]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        return self.pairs[idx]


def evaluate_validation(model: SentenceTransformer, val_pairs: list[dict], device: torch.device) -> dict:
    """Evaluate retrieval metrics (Hit@1, Hit@3, Hit@5, MRR@5) on the 75 validation pairs."""
    model.eval()
    with torch.no_grad():
        queries = [p["query"] for p in val_pairs]
        passages = [p["passage"] for p in val_pairs]

        # Encode queries and passages
        q_embs = model.encode(queries, convert_to_tensor=True, show_progress_bar=False, device=device)
        p_embs = model.encode(passages, convert_to_tensor=True, show_progress_bar=False, device=device)

        q_embs = F.normalize(q_embs, p=2, dim=1)
        p_embs = F.normalize(p_embs, p=2, dim=1)

        # Similarity matrix: (N_val, N_val)
        sim_matrix = torch.mm(q_embs, p_embs.t())

        n = len(val_pairs)
        hit_1 = 0
        hit_3 = 0
        hit_5 = 0
        mrr = 0.0

        for i in range(n):
            scores = sim_matrix[i].cpu().tolist()
            # Rank descending
            ranked_indices = sorted(range(n), key=lambda idx: scores[idx], reverse=True)
            # Target is index i
            if i in ranked_indices[:1]:
                hit_1 += 1
            if i in ranked_indices[:3]:
                hit_3 += 1
            if i in ranked_indices[:5]:
                hit_5 += 1
            if i in ranked_indices[:5]:
                rank = ranked_indices[:5].index(i) + 1
                mrr += 1.0 / rank

        return {
            "val_hit_1": hit_1 / n,
            "val_hit_3": hit_3 / n,
            "val_hit_5": hit_5 / n,
            "val_mrr_5": mrr / n,
        }


def main() -> None:
    print("=" * 80)
    print("  EV DOMAIN-SPECIFIC EMBEDDING ADAPTATION TRAINING")
    print("=" * 80)

    # 1. Load datasets
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_json = json.load(f)
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        val_json = json.load(f)

    train_pairs = train_json["pairs"]
    val_pairs = val_json["pairs"]

    print(f"Loaded {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs.")

    # 2. Setup DataLoader
    def collate_fn(batch):
        return {
            "queries": [item["query"] for item in batch],
            "passages": [item["passage"] for item in batch],
        }

    train_loader = DataLoader(
        PairDataset(train_pairs),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # 3. Load base model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Step 1] Loading base model ({BASE_MODEL_NAME}) on {device}...")
    model = SentenceTransformer(BASE_MODEL_NAME, token=False, device=str(device))
    model.max_seq_length = MAX_SEQ_LENGTH

    # Pre-training baseline validation score
    pre_val = evaluate_validation(model, val_pairs, device)
    print(f"Pre-training Val MRR@5: {pre_val['val_mrr_5']:.4f} | Hit@1: {pre_val['val_hit_1']:.4f} | Hit@3: {pre_val['val_hit_3']:.4f}")

    # 4. Optimizer & Loss setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS

    print(f"\n[Step 2] Starting Training:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  MNRL scale: {MNRL_SCALE}")
    print(f"  Total steps: {total_steps}")

    t_start = time.time()
    best_mrr = pre_val["val_mrr_5"]
    epoch_logs = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        step_count = 0

        for batch in train_loader:
            optimizer.zero_grad()

            q_features = model.tokenize(batch["queries"])
            p_features = model.tokenize(batch["passages"])

            # Move tensor features to device
            q_features = {k: v.to(device) for k, v in q_features.items() if isinstance(v, torch.Tensor)}
            p_features = {k: v.to(device) for k, v in p_features.items() if isinstance(v, torch.Tensor)}

            q_out = model(q_features)
            p_out = model(p_features)

            q_emb = F.normalize(q_out["sentence_embedding"], p=2, dim=1)
            p_emb = F.normalize(p_out["sentence_embedding"], p=2, dim=1)

            # Cosine similarity matrix scaled by MNRL temperature
            # shape: (batch_size, batch_size)
            sim_matrix = torch.mm(q_emb, p_emb.t()) * MNRL_SCALE
            target_labels = torch.arange(len(batch["queries"]), device=device)

            # In-batch cross entropy loss
            loss = F.cross_entropy(sim_matrix, target_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step_count += 1

        avg_loss = total_loss / max(step_count, 1)

        # Validation after epoch
        val_metrics = evaluate_validation(model, val_pairs, device)
        is_best = val_metrics["val_mrr_5"] >= best_mrr
        if is_best:
            best_mrr = val_metrics["val_mrr_5"]
            model.save(str(OUTPUT_MODEL_DIR))

        status_flag = " [BEST MODEL SAVED]" if is_best else ""
        print(f"Epoch {epoch}/{EPOCHS} - Avg Loss: {avg_loss:.4f} | Val MRR@5: {val_metrics['val_mrr_5']:.4f} | Val Hit@1: {val_metrics['val_hit_1']:.4f}{status_flag}")

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            **val_metrics,
            "saved_as_best": is_best,
        })

    elapsed_seconds = round(time.time() - t_start, 2)
    print(f"\n[Step 3] Fine-tuning finished in {elapsed_seconds}s.")
    print(f"Best Validation MRR@5: {best_mrr:.4f} (Saved to {OUTPUT_MODEL_DIR.name})")

    # 5. Write complete training configuration & metadata
    config_record = {
        "base_model": BASE_MODEL_NAME,
        "embedding_dimension": 384,
        "loss_function": "MultipleNegativesRankingLoss (InfoNCE)",
        "mnrl_scale": MNRL_SCALE,
        "similarity_metric": "cosine",
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_seq_length": MAX_SEQ_LENGTH,
        "random_seed": SEED,
        "training_time_seconds": elapsed_seconds,
        "training_pairs_count": len(train_pairs),
        "validation_pairs_count": len(val_pairs),
        "unique_passages_count": train_json["unique_passages"],
        "source_documents_count": len(train_json["source_documents"]),
        "device": str(device),
        "pre_training_val_mrr_5": pre_val["val_mrr_5"],
        "best_val_mrr_5": best_mrr,
        "epoch_history": epoch_logs,
        "output_path": str(OUTPUT_MODEL_DIR),
    }

    with open(OUTPUT_MODEL_DIR / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)

    print("Complete training hyperparameters saved to training_config.json.")


if __name__ == "__main__":
    main()
