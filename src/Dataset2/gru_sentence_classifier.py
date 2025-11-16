"""GRU-based sentence classifier for extractive summarisation on CNN/DailyMail."""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset

if __package__ in {None, ""}:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, ".."))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

from shared_utils import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_SEED,
    DEFAULT_TEST_FILE,
    DEFAULT_TRAIN_FILE,
    DEFAULT_VAL_FILE,
    EarlyStopper,
    build_sentence_dataset,
    build_vocab,
    compute_all_metrics,
    encode_tokens,
    ensure_dir,
    progress_bar,
    save_report,
    segment_sentences,
    set_global_seed,
    tokenize_sentence,
)

STAGE_NAME = "gru_sentence_classifier"
MODEL_PATH = "./models/gru_sentence_classifier.pt"
VOCAB_PATH = "./models/gru_sentence_vocab.json"
METRICS_PATH = "./results/gru_sentence_classifier_metrics.json"
SAMPLES_PATH = "./results/gru_sentence_classifier_samples.json"

TRAIN_ARTICLES = 20_000
VAL_ARTICLES = 2_000
TEST_ARTICLES = 2_000
ROUGE_LABEL_THRESHOLD = 0.35
TOP_K_SENTENCES = 3
DECISION_THRESHOLD = 0.5
MAX_VOCAB = 30_000
MAX_SEQ_LEN = 96
EMBED_DIM = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.2
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 8
EARLY_STOPPING_PATIENCE = 2
MAX_GRAD_NORM = 1.0
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()
RUN_INTERACTIVE = False

DEV_MODE = os.getenv("DATASET2_FAST_DEV_RUN", "0").lower() in {"1", "true", "yes"}
if DEV_MODE:
    TRAIN_ARTICLES = min(TRAIN_ARTICLES, 2_000)
    VAL_ARTICLES = min(VAL_ARTICLES, 400)
    TEST_ARTICLES = min(TEST_ARTICLES, 400)

GLOBAL_SEED = DEFAULT_SEED
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STAGE_CONFIG: Dict[str, object] = {
    "train_articles": TRAIN_ARTICLES,
    "val_articles": VAL_ARTICLES,
    "test_articles": TEST_ARTICLES,
    "rouge_label_threshold": ROUGE_LABEL_THRESHOLD,
    "top_k_sentences": TOP_K_SENTENCES,
    "decision_threshold": DECISION_THRESHOLD,
    "max_vocab": MAX_VOCAB,
    "max_seq_len": MAX_SEQ_LEN,
    "embed_dim": EMBED_DIM,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "dropout": DROPOUT,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "dev_mode": DEV_MODE,
    "device": str(DEVICE),
    "seed": GLOBAL_SEED,
}


class SentenceDataset(Dataset):
    def __init__(self, df, stoi: Dict[str, int], max_len: int):
        self.df = df.reset_index(drop=True)
        self.sentences = self.df["sentence"].tolist()
        self.labels = self.df["label"].astype(int).tolist()
        self.article_ids = self.df["article_id"].tolist()
        self.sentence_indices = self.df["sentence_index"].tolist()
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int):
        sentence = self.sentences[idx]
        tokens = tokenize_sentence(sentence)
        length = min(len(tokens) + 2, self.max_len)
        encoded = encode_tokens(tokens, self.stoi, self.max_len)
        return {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "length": length,
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


def collate_batch(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    labels = torch.stack([item["label"] for item in batch])
    return input_ids, lengths, labels


class GRUSentenceClassifier(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=pad_idx)
        self.gru = nn.GRU(
            input_size=EMBED_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        last_hidden = hidden[-1]
        logits = self.classifier(self.dropout(last_hidden))
        return logits.squeeze(-1)


def build_vocab_from_sentences(df) -> Tuple[Dict[str, int], Dict[int, str]]:
    tokenised = [tokenize_sentence(sent) for sent in df["sentence"].tolist()]
    stoi, itos = build_vocab(tokenised, MAX_VOCAB)
    return stoi, itos


def save_vocab(itos: Dict[int, str], path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    ordered = [itos[idx] for idx in sorted(itos)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"itos": ordered}, f, ensure_ascii=False, indent=2)


def predict_loader(model: nn.Module, loader: DataLoader, criterion: nn.Module | None = None):
    model.eval()
    all_probs: List[float] = []
    all_labels: List[float] = []
    total_loss = 0.0
    with torch.no_grad():
        for input_ids, lengths, labels in loader:
            input_ids = input_ids.to(DEVICE)
            lengths = lengths.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(input_ids, lengths)
            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = (total_loss / len(loader.dataset)) if criterion is not None else None
    return avg_loss, all_probs, all_labels


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
        raise ValueError(f"Probability count mismatch for {name} split")
    df = sentence_df.copy()
    df["prob"] = probs
    preds = (df["prob"] >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        df["label"].astype(int), preds, average="binary", zero_division=0
    )

    grouped = defaultdict(list)
    for row in df.itertuples():
        grouped[row.article_id].append(
            {
                "sentence": row.sentence,
                "prob": row.prob,
                "sentence_index": row.sentence_index,
            }
        )

    article_map = {record["article_id"]: record for record in article_records}
    summaries: List[str] = []
    references: List[str] = []
    samples: List[Dict[str, object]] = []

    for article_id, entries in grouped.items():
        ranked = sorted(entries, key=lambda item: (-item["prob"], item["sentence_index"]))
        chosen = ranked[:top_k]
        ordered = sorted(chosen, key=lambda item: item["sentence_index"])
        summary = " ".join(item["sentence"] for item in ordered)
        highlight_text = article_map[article_id]["highlights"]
        summaries.append(summary)
        references.append(highlight_text)
        if len(samples) < sample_cap:
            samples.append(
                {
                    "article_id": article_id,
                    "highlights": highlight_text,
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
    """Grid-search a sigmoid cutoff that maximises validation F1."""
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


def train_epoch(model, loader, criterion, optimizer, epoch: int):
    model.train()
    total_loss = 0.0
    for input_ids, lengths, labels in progress_bar(loader, desc=f"Epoch {epoch} training"):
        input_ids = input_ids.to(DEVICE)
        lengths = lengths.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


def interactive_cli(model: nn.Module, stoi: Dict[str, int]):
    print("Interactive GRU sentence classifier — type 'exit' to quit.")
    while True:
        article = input("Article> ").strip()
        if article.lower() in {"", "exit", "quit"}:
            break
        sentences = segment_sentences(article)
        if not sentences:
            print("No sentences detected.\n")
            continue
        encoded = []
        lengths = []
        for sent in sentences:
            tokens = tokenize_sentence(sent)
            lengths.append(min(len(tokens) + 2, MAX_SEQ_LEN))
            encoded.append(torch.tensor(encode_tokens(tokens, stoi, MAX_SEQ_LEN), dtype=torch.long))
        batch = torch.stack(encoded).to(DEVICE)
        length_tensor = torch.tensor(lengths, dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            probs = torch.sigmoid(model(batch, length_tensor)).cpu().tolist()
        ranked = sorted(
            [
                {"sentence": sent, "prob": prob, "sentence_index": idx}
                for idx, (sent, prob) in enumerate(zip(sentences, probs))
            ],
            key=lambda item: (-item["prob"], item["sentence_index"]),
        )
        chosen = ranked[:TOP_K_SENTENCES]
        chosen_sorted = sorted(chosen, key=lambda item: item["sentence_index"])
        summary = " ".join(item["sentence"] for item in chosen_sorted)
        print("\nTop sentences:\n", summary or "<none>", "\n")


def main() -> None:
    set_global_seed(GLOBAL_SEED)
    ensure_dir("./models")
    ensure_dir("./results")

    start_time = time.time()
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

    stoi, itos = build_vocab_from_sentences(train_df)
    save_vocab(itos, VOCAB_PATH)

    train_dataset = SentenceDataset(train_df, stoi, MAX_SEQ_LEN)
    val_dataset = SentenceDataset(val_df, stoi, MAX_SEQ_LEN)
    test_dataset = SentenceDataset(test_df, stoi, MAX_SEQ_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_batch,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_batch,
    )

    pos_count = float(train_df["label"].sum())
    neg_count = float(len(train_df) - pos_count)
    if pos_count == 0:
        pos_weight_value = 1.0
    else:
        pos_weight_value = max(neg_count / pos_count, 1.0)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float, device=DEVICE)
    print(f"Using positive-class weight: {pos_weight_value:.3f}")

    model = GRUSentenceClassifier(len(stoi), stoi["<pad>"]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0
    stopper = EarlyStopper(patience=EARLY_STOPPING_PATIENCE)
    train_history: List[Dict[str, float]] = []
    val_history: List[Dict[str, float]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_probs, val_labels = predict_loader(model, val_loader, criterion)
        val_preds = [1 if p >= DECISION_THRESHOLD else 0 for p in val_probs]
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average="binary", zero_division=0
        )
        train_history.append({"epoch": epoch, "loss": train_loss})
        val_history.append(
            {
                "epoch": epoch,
                "loss": val_loss,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, MODEL_PATH)

        print(
            f"[Epoch {epoch}/{MAX_EPOCHS}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}",
            flush=True,
        )

        if stopper.step(f1):
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_state)

    train_loss, train_probs, train_labels = predict_loader(model, train_eval_loader, criterion)
    val_loss, val_probs, val_labels = predict_loader(model, val_loader, criterion)
    test_loss, test_probs, _ = predict_loader(model, test_loader, criterion)

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
        json.dump(
            {
                "train": {**train_metrics, "loss": train_loss},
                "validation": {**val_metrics, "loss": val_loss},
                "test": {**test_metrics, "loss": test_loss},
            },
            f,
            indent=2,
        )
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

    print(f"Saved GRU classifier to {MODEL_PATH}, vocab to {VOCAB_PATH}, and report to {report_path}.")

    if RUN_INTERACTIVE:
        interactive_cli(model, stoi)


if __name__ == "__main__":
    main()
