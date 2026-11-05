import os
import time

# Force offline mode for local cached weights
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util

print("=" * 80)
print("  EV FAULT DIAGNOSIS RAG - PILOT EMBEDDING FINE-TUNING EXPERIMENT")
print("=" * 80)

# 1. Load base pre-trained model
print("\n[Step 1] Loading base model (sentence-transformers/all-MiniLM-L6-v2)...", flush=True)
t_start = time.time()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
print(f"Base model loaded in {time.time() - t_start:.2f}s", flush=True)

# 2. Pilot EV Dataset (from retrieval_eval.json & retrieval_baseline.md)
pilot_dataset = [
    {
        "id": "charging-compatible-700w",
        "query": "Can I use my existing 700W charger with the Ather 450S, and how long will recharging take?",
        "positive": "Of course you can use the old 700W charger as it is compatible with all new Ather scooters. It will take 4 hr 30 min to charge from 0 to 100% and 3 hr 30 min to charge from 0%-80%."
    },
    {
        "id": "charging-time-450s",
        "query": "What home charging time is listed for the Ather 450S?",
        "positive": "Home charging time for Ather 450S is approximately 5 hours 45 minutes for 0-80% using the standard portable charger."
    },
    {
        "id": "fallsafe-rider-fall",
        "query": "What does FallSafe do when the rider falls?",
        "positive": "FallSafe automatically cuts off motor power and turns off the indicator lights to prevent unintentional acceleration in case the vehicle falls down or tilts past the threshold."
    },
    {
        "id": "rsa-discharged-battery",
        "query": "What roadside assistance is described when a fully discharged battery leaves the vehicle immovable?",
        "positive": "If the vehicle is completely discharged, Roadside Assistance provides towing service to the nearest charging point or Ather Grid."
    },
    {
        "id": "eight70-battery-degradation",
        "query": "What battery degradation protection is described by Eight70 Warranty?",
        "positive": "Eight70 Warranty provides coverage guaranteeing that battery health will not drop below 70% state of health within 8 years or 80,000 km."
    }
]

# Distractor passage (unrelated vehicle feature)
distractor_chunk = "Bluetooth pairing requires turning on vehicle ignition and confirming PIN on the dashboard display screen."

# 3. Baseline Evaluation (Before Fine-Tuning)
print("\n[Step 2] Evaluating Baseline (Pre-Trained) Embeddings...", flush=True)
model.eval()
before_scores = []
for item in pilot_dataset:
    q_emb = model.encode(item["query"], convert_to_tensor=True)
    pos_emb = model.encode(item["positive"], convert_to_tensor=True)
    neg_emb = model.encode(distractor_chunk, convert_to_tensor=True)
    
    pos_sim = util.cos_sim(q_emb, pos_emb).item()
    neg_sim = util.cos_sim(q_emb, neg_emb).item()
    margin = pos_sim - neg_sim
    before_scores.append({
        "id": item["id"],
        "pos_sim": pos_sim,
        "neg_sim": neg_sim,
        "margin": margin
    })

# 4. Direct PyTorch Fine-Tuning with InfoNCE / Multiple Negatives Ranking Loss
print("\n[Step 3] Fine-tuning model on pilot EV chunks (InfoNCE Loss)...", flush=True)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

queries = [item["query"] for item in pilot_dataset]
positives = [item["positive"] for item in pilot_dataset]

t_train_start = time.time()
epochs = 8
scale = 20.0  # Temperature scale for contrastive embeddings

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    
    # Preprocess and forward pass through Transformer + Pooling
    feat_q = model.preprocess(queries)
    feat_p = model.preprocess(positives)
    
    emb_q = model(feat_q)["sentence_embedding"]
    emb_p = model(feat_p)["sentence_embedding"]
    
    # L2 Normalization
    q_norm = F.normalize(emb_q, p=2, dim=1)
    p_norm = F.normalize(emb_p, p=2, dim=1)
    
    # Similarity matrix: (N x N) where diagonal is positive pairs, off-diagonal are in-batch negatives
    similarity_matrix = torch.mm(q_norm, p_norm.transpose(0, 1)) * scale
    targets = torch.arange(len(queries), dtype=torch.long)
    
    loss = loss_fn(similarity_matrix, targets)
    loss.backward()
    optimizer.step()
    
    if epoch % 2 == 0 or epoch == epochs:
        print(f"  Epoch {epoch}/{epochs} | InfoNCE Loss: {loss.item():.4f}", flush=True)

train_duration = time.time() - t_train_start
print(f"Training converged in {train_duration:.2f} seconds!", flush=True)

# 5. Post-Finetuning Evaluation
print("\n[Step 4] Evaluating Fine-Tuned Model...", flush=True)
model.eval()
after_scores = []
for item in pilot_dataset:
    q_emb = model.encode(item["query"], convert_to_tensor=True)
    pos_emb = model.encode(item["positive"], convert_to_tensor=True)
    neg_emb = model.encode(distractor_chunk, convert_to_tensor=True)
    
    pos_sim = util.cos_sim(q_emb, pos_emb).item()
    neg_sim = util.cos_sim(q_emb, neg_emb).item()
    margin = pos_sim - neg_sim
    after_scores.append({
        "id": item["id"],
        "pos_sim": pos_sim,
        "neg_sim": neg_sim,
        "margin": margin
    })

# 6. Print Comparison Table
print("\n" + "=" * 80)
print(f"{'Pilot Query ID':<28} | {'Pos (Pre)':<10} | {'Pos (Post)':<10} | {'Gain':<7} | {'Margin (Pre)':<12} | {'Margin (Post)'}")
print("-" * 80)
total_pre = 0.0
total_post = 0.0
total_m_pre = 0.0
total_m_post = 0.0

for b, a in zip(before_scores, after_scores):
    gain = a["pos_sim"] - b["pos_sim"]
    margin_gain = a["margin"] - b["margin"]
    total_pre += b["pos_sim"]
    total_post += a["pos_sim"]
    total_m_pre += b["margin"]
    total_m_post += a["margin"]
    print(f"{b['id']:<28} | {b['pos_sim']:<10.4f} | {a['pos_sim']:<10.4f} | {gain:+6.4f} | {b['margin']:<12.4f} | {a['margin']:<12.4f} ({margin_gain:+6.4f})")

avg_pre = total_pre / len(before_scores)
avg_post = total_post / len(after_scores)
avg_m_pre = total_m_pre / len(before_scores)
avg_m_post = total_m_post / len(after_scores)
print("-" * 80)
print(f"{'AVERAGE':<28} | {avg_pre:<10.4f} | {avg_post:<10.4f} | {avg_post - avg_pre:+6.4f} | {avg_m_pre:<12.4f} | {avg_m_post:<12.4f} ({avg_m_post - avg_m_pre:+6.4f})")
print("=" * 80)

# 7. Save model
save_path = os.path.join(os.path.dirname(__file__), "data", "pilot_minilm_ev")
os.makedirs(save_path, exist_ok=True)
model.save(save_path)
print(f"\n[Step 5] Fine-tuned model saved successfully to: {save_path}")

# 8. Save results to JSON
import json
json_path = os.path.join(os.path.dirname(__file__), "data", "pilot_finetune_results.json")
result_payload = {
    "experiment_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "base_model": "sentence-transformers/all-MiniLM-L6-v2",
    "saved_model_path": save_path,
    "training_hyperparameters": {
        "loss": "InfoNCE / MultipleNegativesRankingLoss",
        "optimizer": "AdamW",
        "learning_rate": 2e-5,
        "epochs": epochs,
        "scale": scale,
        "duration_seconds": round(train_duration, 2)
    },
    "results": [
        {
            "id": b["id"],
            "pre_similarity": round(b["pos_sim"], 4),
            "post_similarity": round(a["pos_sim"], 4),
            "gain": round(a["pos_sim"] - b["pos_sim"], 4),
            "pre_margin": round(b["margin"], 4),
            "post_margin": round(a["margin"], 4),
            "margin_gain": round(a["margin"] - b["margin"], 4)
        }
        for b, a in zip(before_scores, after_scores)
    ],
    "summary": {
        "average_pre_similarity": round(avg_pre, 4),
        "average_post_similarity": round(avg_post, 4),
        "average_similarity_gain": round(avg_post - avg_pre, 4),
        "average_pre_margin": round(avg_m_pre, 4),
        "average_post_margin": round(avg_m_post, 4),
        "average_margin_gain": round(avg_m_post - avg_m_pre, 4)
    }
}
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result_payload, f, indent=2)
print(f"Results JSON saved to: {json_path}")
print("Experiment complete!")
