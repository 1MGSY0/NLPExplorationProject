"""BERT-based sentence classifier for extractive summarisation."""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import (  
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

if __package__ in {None, ""}:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, ".."))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

from shared_utils import ( 
    DEFAULT_DATA_DIR,
    DEFAULT_SEED,
    DEFAULT_TEST_FILE,
    DEFAULT_TRAIN_FILE,
    DEFAULT_VAL_FILE,
    build_sentence_dataset,
    compute_all_metrics,
    ensure_dir,
    progress_bar,
    save_report,
    segment_sentences,
    set_global_seed,
)

STAGE_NAME = "bert_sentence_classifier"
MODEL_PATH = "./models/bert_sentence_classifier.pt"
METRICS_PATH = "./results/bert_sentence_classifier_metrics.json"
SAMPLES_PATH = "./results/bert_sentence_classifier_samples.json"

BASE_MODEL_NAME = os.getenv("DATASET2_BERT_BACKBONE", "distilbert-base-uncased")
TRAIN_ARTICLES = 2_000
VAL_ARTICLES = 2_000
TEST_ARTICLES = 2_000
ROUGE_LABEL_THRESHOLD = 0.35
TOP_K_SENTENCES = 3
DECISION_THRESHOLD = 0.5
MAX_SEQ_LEN = 160
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 3
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION_STEPS = 2
MAX_GRAD_NORM = 1.0
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()
RUN_INTERACTIVE = False

DEV_MODE = os.getenv("DATASET2_FAST_DEV_RUN", "0").lower() in {"1", "true", "yes"}
if DEV_MODE:
    TRAIN_ARTICLES = min(TRAIN_ARTICLES, 2_000)
    VAL_ARTICLES = min(VAL_ARTICLES, 400)
    TEST_ARTICLES = min(TEST_ARTICLES, 400)
    MAX_EPOCHS = 1

GLOBAL_SEED = DEFAULT_SEED
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STAGE_CONFIG: Dict[str, object] = {
    "backbone": BASE_MODEL_NAME,
    "train_articles": TRAIN_ARTICLES,
    "val_articles": VAL_ARTICLES,
    "test_articles": TEST_ARTICLES,
    "rouge_label_threshold": ROUGE_LABEL_THRESHOLD,
    "top_k_sentences": TOP_K_SENTENCES,
    "decision_threshold": DECISION_THRESHOLD,
    "max_seq_len": MAX_SEQ_LEN,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "max_epochs": MAX_EPOCHS,
    "warmup_ratio": WARMUP_RATIO,
    "grad_accum": GRADIENT_ACCUMULATION_STEPS,
    "dev_mode": DEV_MODE,
    "device": str(DEVICE),
    "seed": GLOBAL_SEED,
}


@dataclass
class SentenceBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class SentenceDataset(Dataset):
    def __init__(self, df, tokenizer, max_len: int):
        self.df = df.reset_index(drop=True)
        self.sentences = self.df["sentence"].tolist()
        self.labels = self.df["label"].astype(int).tolist()
        self.article_ids = self.df["article_id"].tolist()
        self.sentence_indices = self.df["sentence_index"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int) -> SentenceBatch:
        sentence = self.sentences[idx]
        encoding = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return SentenceBatch(input_ids, attention_mask, label)


def collate_fn(batch: List[SentenceBatch]):
    input_ids = torch.stack([item.input_ids for item in batch])
    attention_mask = torch.stack([item.attention_mask for item in batch])
    labels = torch.stack([item.labels for item in batch])
    return input_ids, attention_mask, labels


def predict(loader: DataLoader, model) -> Tuple[List[float], List[float]]:
    model.eval()
    probs: List[float] = []
    labels: List[float] = []
    with torch.no_grad():
        for input_ids, attention_mask, batch_labels in loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            batch_probs = torch.sigmoid(logits)
            probs.extend(batch_probs.cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
    return probs, labels


def evaluate_split(
    name: str,
    sentence_df,
    article_records,
    probs: List[float],
    top_k: int,
    threshold: float,
    sample_cap: int = 10,
):
    if len(probs) != len(sentence_df):
        raise ValueError(f"{name} probs length mismatch")
    df = sentence_df.copy()
    df["prob"] = probs
    preds = (df["prob"] >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        df["label"].astype(int), preds, average="binary", zero_division=0
    )

    grouped = {}
    for row in df.itertuples():
        grouped.setdefault(row.article_id, []).append(
            {"sentence": row.sentence, "prob": row.prob, "sentence_index": row.sentence_index}
        )

    article_map = {record["article_id"]: record for record in article_records}
    summaries: List[str] = []
    references: List[str] = []
    samples: List[Dict[str, object]] = []
    for article_id, entries in grouped.items():
        ranked = sorted(entries, key=lambda item: (-item["prob"], item["sentence_index"]))
        ordered = sorted(ranked[:top_k], key=lambda item: item["sentence_index"])
        summary = " ".join(item["sentence"] for item in ordered)
        highlight = article_map[article_id]["highlights"]
        summaries.append(summary)
        references.append(highlight)
        if len(samples) < sample_cap:
            samples.append(
                {
                    "article_id": article_id,
                    "highlights": highlight,
                    "predicted_summary": summary,
                    "top_sentence_indices": [item["sentence_index"] for item in ordered],
                    "probabilities": [item["prob"] for item in ordered],
                }
            )

    rouge_metrics = compute_all_metrics(summaries, references)
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        **rouge_metrics,
    }
    return metrics, samples


def find_best_threshold(
    probs: List[float],
    labels: List[float],
    default_threshold: float = DECISION_THRESHOLD,
    step: float = 0.05,
) -> float:
    if not probs:
        return default_threshold
    thresholds = {default_threshold}
    thresh = step
    while thresh < 1.0:
        thresholds.add(round(thresh, 3))
        thresh += step
    best_threshold = default_threshold
    best_f1 = -1.0
    for threshold in sorted(thresholds):
        preds = [1 if p >= threshold else 0 for p in probs]
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


def interactive_cli(model, tokenizer):
    print("Interactive BERT classifier — type 'exit' to quit.")
    while True:
        article = input("Article> ").strip()
        if article.lower() in {"", "exit", "quit"}:
            break
        sentences = segment_sentences(article)
        encoded = tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().tolist()
        ranked = sorted(
            [
                {"sentence": sent, "prob": prob, "sentence_index": idx}
                for idx, (sent, prob) in enumerate(zip(sentences, probs))
            ],
            key=lambda item: (-item["prob"], item["sentence_index"]),
        )
        ordered = sorted(ranked[:TOP_K_SENTENCES], key=lambda item: item["sentence_index"])
        summary = " ".join(item["sentence"] for item in ordered)
        print("\nTop sentences:\n", summary or "<none>", "\n")


def main() -> None:
    set_global_seed(GLOBAL_SEED)
    ensure_dir("./models")
    ensure_dir("./results")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=1,
        problem_type="single_label_classification",
    ).to(DEVICE)

    train_df, train_articles = build_sentence_dataset(
        "train",
        n_articles=TRAIN_ARTICLES,
        rouge_threshold=ROUGE_LABEL_THRESHOLD,
        data_dir=DEFAULT_DATA_DIR,
        train_file=DEFAULT_TRAIN_FILE,
        val_file=DEFAULT_VAL_FILE,
        test_file=DEFAULT_TEST_FILE,
        seed=GLOBAL_SEED,
    )
    val_df, val_articles = build_sentence_dataset(
        "val",
        n_articles=VAL_ARTICLES,
        rouge_threshold=ROUGE_LABEL_THRESHOLD,
        data_dir=DEFAULT_DATA_DIR,
        train_file=DEFAULT_TRAIN_FILE,
        val_file=DEFAULT_VAL_FILE,
        test_file=DEFAULT_TEST_FILE,
        seed=GLOBAL_SEED,
    )
    test_df, test_articles = build_sentence_dataset(
        "test",
        n_articles=TEST_ARTICLES,
        rouge_threshold=ROUGE_LABEL_THRESHOLD,
        data_dir=DEFAULT_DATA_DIR,
        train_file=DEFAULT_TRAIN_FILE,
        val_file=DEFAULT_VAL_FILE,
        test_file=DEFAULT_TEST_FILE,
        seed=GLOBAL_SEED,
    )

    pos_count = float(train_df["label"].sum())
    neg_count = float(len(train_df) - pos_count)
    if pos_count == 0:
        pos_weight_value = 1.0
    else:
        pos_weight_value = max(neg_count / pos_count, 1.0)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float, device=DEVICE)
    print(f"Using positive-class weight: {pos_weight_value:.3f}")

    train_dataset = SentenceDataset(train_df, tokenizer, MAX_SEQ_LEN)
    val_dataset = SentenceDataset(val_df, tokenizer, MAX_SEQ_LEN)
    test_dataset = SentenceDataset(test_df, tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
    )

    total_steps = (len(train_loader) * MAX_EPOCHS) // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_val_f1 = 0.0
    train_history: List[Dict[str, float]] = []
    val_history: List[Dict[str, float]] = []

    start_time = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (input_ids, attention_mask, labels) in enumerate(progress_bar(train_loader, desc=f"Epoch {epoch}"), 1):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            loss = criterion(logits, labels)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            if step % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item() * labels.size(0)
        avg_train_loss = running_loss / len(train_loader.dataset)

        val_probs, val_labels = predict(val_loader, model)
        val_preds = [1 if p >= DECISION_THRESHOLD else 0 for p in val_probs]
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average="binary", zero_division=0
        )
        train_history.append({"epoch": epoch, "loss": avg_train_loss})
        val_history.append(
            {
                "epoch": epoch,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = model.state_dict()
            torch.save(best_state, MODEL_PATH)

        print(
            f"[Epoch {epoch}/{MAX_EPOCHS}] train_loss={avg_train_loss:.4f} "
            f"precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_probs, train_labels = predict(train_loader, model)
    val_probs, val_labels = predict(val_loader, model)
    test_probs, _ = predict(test_loader, model)

    optimized_threshold = find_best_threshold(val_probs, val_labels, DECISION_THRESHOLD)
    STAGE_CONFIG["decision_threshold"] = optimized_threshold
    print(f"Optimized validation threshold: {optimized_threshold:.3f}")

    train_metrics, _ = evaluate_split(
        "Train",
        train_df,
        train_articles,
        train_probs,
        TOP_K_SENTENCES,
        optimized_threshold,
    )
    val_metrics, val_samples = evaluate_split(
        "Validation",
        val_df,
        val_articles,
        val_probs,
        TOP_K_SENTENCES,
        optimized_threshold,
    )
    test_metrics, test_samples = evaluate_split(
        "Test",
        test_df,
        test_articles,
        test_probs,
        TOP_K_SENTENCES,
        optimized_threshold,
    )

    elapsed = time.time() - start_time

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"train": train_metrics, "validation": val_metrics, "test": test_metrics}, f, indent=2)
    with open(SAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump({"validation": val_samples, "test": test_samples}, f, indent=2)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    report_path = save_report(
        STAGE_NAME,
        train_history,
        val_history,
        test_metrics,
        test_samples,
        training_time=elapsed,
        config=STAGE_CONFIG,
        output_dir="./results",
        run_id=run_id,
    )
    print(f"Saved BERT classifier to {MODEL_PATH} and report to {report_path}.")

    if RUN_INTERACTIVE:
        interactive_cli(model, tokenizer)


if __name__ == "__main__":
    main()
