"""Common utilities for Dataset2 stages (config, data, metrics, reporting)."""
from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import asdict, is_dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

try:  # Optional torch seeding support
    import torch
except ImportError:  # pragma: no cover - torch unavailable in classical stage
    torch = None

try:
    from nltk.translate.meteor_score import meteor_score
except ImportError as exc:  # pragma: no cover - surfaced during runtime
    raise RuntimeError(
        "nltk is required for METEOR computation. Install it via 'pip install nltk'."
    ) from exc


DEFAULT_DATA_DIR = "../../data/Dataset2"
DEFAULT_TRAIN_FILE = "train.csv"
DEFAULT_VAL_FILE = "validation.csv"
DEFAULT_TEST_FILE = "test.csv"
DEFAULT_SEED = 42
DEFAULT_LENGTH_BUCKETS: Tuple[Tuple[int, int, str], ...] = (
    (0, 40, "short"),
    (40, 80, "medium"),
    (80, 160, "long"),
    (160, math.inf, "xl"),
)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy (if installed), and torch (if available)."""

    random.seed(seed)
    try:  # pragma: no cover - numpy optional
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


_CLEAN_RE = re.compile(r"[^a-z0-9.,!?;:'\"()\-\s]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    text = str(text).lower().replace("\n", " ").replace("\r", " ")
    text = _CLEAN_RE.sub(" ", text)
    return normalize_whitespace(text)


def segment_sentences(text: str) -> List[str]:
    cleaned = clean_text(text)
    parts = _SENT_SPLIT_RE.split(cleaned)
    return [normalize_whitespace(p) for p in parts if normalize_whitespace(p)]


def tokenize(text: str) -> List[str]:
    cleaned = clean_text(text)
    return cleaned.split() if cleaned else []


def tokenize_sentence(sentence: str) -> List[str]:
    return tokenize(sentence)


def bucket_for_length(length: int, buckets: Tuple[Tuple[int, int, str], ...]) -> str:
    for lower, upper, name in buckets:
        if lower <= length < upper:
            return name
    return buckets[-1][2]


def _stratified_sample(
    df: pd.DataFrame,
    desired_n: int,
    buckets: Tuple[Tuple[int, int, str], ...],
    seed: int,
) -> pd.DataFrame:
    n = min(desired_n, len(df))
    if n <= 0:
        return df.iloc[0:0]

    working = df.copy()
    working["_bucket"] = working["highlights"].apply(lambda x: bucket_for_length(len(tokenize(x)), buckets))
    per_bucket = max(1, n // len(buckets))
    sampled: List[pd.DataFrame] = []
    taken: set[int] = set()
    remaining = n

    for _, _, bucket_name in buckets:
        bucket_df = working[working["_bucket"] == bucket_name]
        if bucket_df.empty:
            continue
        take = min(per_bucket, remaining, len(bucket_df))
        if take <= 0:
            continue
        part = bucket_df.sample(n=take, random_state=seed, replace=False)
        sampled.append(part)
        taken.update(part.index.tolist())
        remaining -= take
        if remaining <= 0:
            break

    if remaining > 0:
        leftover = working.loc[~working.index.isin(list(taken))]
        if not leftover.empty:
            sampled.append(leftover.sample(n=min(remaining, len(leftover)), random_state=seed))

    if not sampled:
        sampled.append(working.sample(n=n, random_state=seed))

    result = pd.concat(sampled)
    return result.drop(columns=["_bucket"], errors="ignore").sample(frac=1, random_state=seed)


def load_dataframe(
    split: str,
    n_samples: Optional[int] = None,
    balanced: bool = False,
    data_dir: str = DEFAULT_DATA_DIR,
    train_file: str = DEFAULT_TRAIN_FILE,
    val_file: str = DEFAULT_VAL_FILE,
    test_file: str = DEFAULT_TEST_FILE,
    seed: int = DEFAULT_SEED,
    length_buckets: Tuple[Tuple[int, int, str], ...] = DEFAULT_LENGTH_BUCKETS,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Load and preprocess a CSV split, optionally sampling/balancing rows."""

    file_map = {
        "train": train_file,
        "val": val_file,
        "validation": val_file,
        "test": test_file,
    }
    filename = file_map.get(split, split)
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path).dropna(subset=["highlights", "article"]).copy()
    df["highlights"] = df["highlights"].apply(clean_text)
    df["article"] = df["article"].apply(clean_text)
    total = len(df)
    desired = n_samples if n_samples is not None and n_samples > 0 else total
    if desired >= total:
        subset = df.reset_index(drop=True)
        meta = {"total_rows": total, "sampled_rows": len(subset), "balanced": False}
        return subset, meta

    if balanced:
        subset = _stratified_sample(df, desired, length_buckets, seed)
    else:
        subset = df.sample(n=desired, random_state=seed)
    subset = subset.reset_index(drop=True)
    meta = {"total_rows": total, "sampled_rows": len(subset), "balanced": balanced}
    return subset, meta


def build_vocab(token_seqs: List[List[str]], max_vocab: int) -> Tuple[Dict[str, int], Dict[int, str]]:
    freq: Dict[str, int] = {}
    for seq in token_seqs:
        for tok in seq:
            freq[tok] = freq.get(tok, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    vocab_tokens = [tok for tok, _ in sorted_tokens[: max_vocab - 4]]
    specials = ["<pad>", "<s>", "</s>", "<unk>"]
    vocab = specials + vocab_tokens
    stoi = {tok: idx for idx, tok in enumerate(vocab)}
    itos = {idx: tok for tok, idx in stoi.items()}
    return stoi, itos


def encode_tokens(tokens: List[str], stoi: Dict[str, int], max_len: int) -> List[int]:
    ids = [stoi.get(tok, stoi["<unk>"]) for tok in tokens[: max_len - 2]]
    pad_needed = max(0, max_len - len(ids) - 2)
    return [stoi["<s>"]] + ids + [stoi["</s>"]] + [stoi["<pad>"]] * pad_needed


def encode_text(text: str, stoi: Dict[str, int], max_len: int) -> List[int]:
    return encode_tokens(tokenize(text), stoi, max_len)


def decode_ids(ids, itos: Dict[int, str]) -> str:
    tokens: List[str] = []
    for idx in ids.tolist() if hasattr(ids, "tolist") else ids:
        tok = itos.get(int(idx), "<unk>")
        if tok == "</s>":
            break
        if tok in {"<pad>", "<s>"}:
            continue
        tokens.append(tok)
    return " ".join(tokens)


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    if len(tokens) < n or n <= 0:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _lcs_length(x: List[str], y: List[str]) -> int:
    dp = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l_f1_tokens(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    prec = lcs / max(1, len(pred_tokens))
    rec = lcs / max(1, len(ref_tokens))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _rouge_score(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    pred_counts = _ngram_counts(pred_tokens, n)
    ref_counts = _ngram_counts(ref_tokens, n)
    if not pred_counts or not ref_counts:
        return 0.0
    overlap = sum(min(count, ref_counts.get(gram, 0)) for gram, count in pred_counts.items())
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, sum(pred_counts.values()))
    recall = overlap / max(1, sum(ref_counts.values()))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _bleu(pred_tokens_list: List[List[str]], ref_tokens_list: List[List[str]], max_n: int = 4) -> float:
    weights = [1.0 / max_n] * max_n
    precisions: List[float] = []
    for n in range(1, max_n + 1):
        overlap = 0
        total = 0
        for pred_tokens, ref_tokens in zip(pred_tokens_list, ref_tokens_list):
            pred_counts = _ngram_counts(pred_tokens, n)
            ref_counts = _ngram_counts(ref_tokens, n)
            overlap += sum(min(count, ref_counts.get(gram, 0)) for gram, count in pred_counts.items())
            total += max(1, len(pred_tokens) - n + 1) if len(pred_tokens) >= n else 0
        precisions.append(max(1e-9, overlap / max(1, total)))
    log_precision = sum(w * math.log(p) for w, p in zip(weights, precisions))
    pred_len = sum(len(tokens) for tokens in pred_tokens_list)
    ref_len = sum(len(tokens) for tokens in ref_tokens_list)
    if pred_len == 0:
        return 0.0
    bp = 1.0 if pred_len > ref_len else math.exp(1 - (ref_len / max(pred_len, 1)))
    return bp * math.exp(log_precision)


def _meteor(pred_sentence: str, ref_sentence: str) -> float:
    return float(meteor_score([ref_sentence.split()], pred_sentence.split()))


def compute_all_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    filtered = [(p.strip(), r.strip()) for p, r in zip(preds, refs) if p.strip() and r.strip()]
    if not filtered:
        return {"bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "meteor": 0.0}
    pred_tokens_list = [p.split() for p, _ in filtered]
    ref_tokens_list = [r.split() for _, r in filtered]
    bleu = _bleu(pred_tokens_list, ref_tokens_list) * 100
    rouge1 = sum(_rouge_score(p, r, 1) for p, r in zip(pred_tokens_list, ref_tokens_list)) / len(filtered)
    rouge2 = sum(_rouge_score(p, r, 2) for p, r in zip(pred_tokens_list, ref_tokens_list)) / len(filtered)
    rougeL = 0.0
    for p, r in zip(pred_tokens_list, ref_tokens_list):
        lcs = _lcs_length(p, r)
        prec = lcs / max(1, len(p))
        rec = lcs / max(1, len(r))
        rougeL += 0.0 if prec + rec == 0 else (2 * prec * rec) / (prec + rec)
    rougeL /= len(filtered)
    meteor_avg = sum(_meteor(p, r) for p, r in zip(preds, refs)) / len(filtered)
    return {
        "bleu": bleu,
        "rouge1": rouge1 * 100,
        "rouge2": rouge2 * 100,
        "rougeL": rougeL * 100,
        "meteor": meteor_avg * 100,
    }


def align_sentences_with_highlights(
    article_text: str,
    highlight_text: str,
    threshold: float = 0.35,
) -> List[Tuple[str, int]]:
    sentences = segment_sentences(article_text)
    highlight_sentences = segment_sentences(highlight_text)
    highlight_tokens = [tokenize_sentence(sent) for sent in highlight_sentences if sent]
    aligned: List[Tuple[str, int]] = []
    for sentence in sentences:
        sent_tokens = tokenize_sentence(sentence)
        best = 0.0
        for ref_tokens in highlight_tokens:
            score = rouge_l_f1_tokens(sent_tokens, ref_tokens)
            if score > best:
                best = score
        label = 1 if best >= threshold else 0
        aligned.append((sentence, label))
    return aligned


def build_sentence_dataset(
    split: str,
    n_articles: Optional[int] = None,
    rouge_threshold: float = 0.35,
    data_dir: str = DEFAULT_DATA_DIR,
    train_file: str = DEFAULT_TRAIN_FILE,
    val_file: str = DEFAULT_VAL_FILE,
    test_file: str = DEFAULT_TEST_FILE,
    seed: int = DEFAULT_SEED,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    df, meta = load_dataframe(
        split,
        n_samples=n_articles,
        balanced=False,
        data_dir=data_dir,
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
        seed=seed,
    )
    samples: List[Dict[str, object]] = []
    article_records: List[Dict[str, object]] = []
    for idx, row in df.iterrows():
        article_id = row.get("id", f"{split}_{idx}")
        aligned = align_sentences_with_highlights(row["article"], row["highlights"], threshold=rouge_threshold)
        sentences = [sent for sent, _ in aligned]
        labels = [label for _, label in aligned]
        article_records.append(
            {
                "article_id": article_id,
                "highlights": row["highlights"],
                "sentences": sentences,
                "labels": labels,
            }
        )
        for sent_idx, (sentence, label) in enumerate(aligned):
            samples.append(
                {
                    "article_id": article_id,
                    "sentence": sentence,
                    "label": int(label),
                    "sentence_index": sent_idx,
                    "split": split,
                }
            )
    dataset_df = pd.DataFrame(samples)
    return dataset_df, article_records


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _serialize_config(config_obj: Optional[object]) -> Optional[Dict[str, object]]:
    if config_obj is None:
        return None
    if isinstance(config_obj, dict):
        return config_obj
    if is_dataclass(config_obj):
        return asdict(config_obj)
    if hasattr(config_obj, "__dict__"):
        payload = {k: v for k, v in vars(config_obj).items() if not k.startswith("_")}
        return payload or {"repr": repr(config_obj)}
    return {"value": repr(config_obj)}


def save_report(
    stage_name: str,
    train_history: List[Dict],
    val_history: List[Dict],
    test_metrics: Dict[str, float],
    samples: List[Dict[str, str]],
    training_time: float,
    config: Optional[Dict[str, object]] = None,
    output_dir: str = "results",
    run_id: Optional[str] = None,
) -> str:
    ensure_dir(output_dir)
    report = {
        "stage": stage_name,
        "config": _serialize_config(config),
        "training_time_seconds": training_time,
        "train_history": train_history,
        "val_history": val_history,
        "test_metrics": test_metrics,
        "samples": samples,
    }
    suffix = f"_{run_id}" if run_id else ""
    path = os.path.join(output_dir, f"{stage_name}{suffix}_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


def progress_bar(iterable: Iterable, desc: str = "") -> Iterable:
    return tqdm(iterable, desc=desc, leave=False)


class EarlyStopper:
    def __init__(self, patience: int = 2, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -math.inf
        self.counter = 0

    def step(self, metric: float) -> bool:
        if metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def cli_inference_loop(
    prompt: str,
    encode_fn: Callable[[str], List[int]],
    generate_fn: Callable[[List[int]], str],
    exit_tokens: Tuple[str, ...] = ("exit", "quit", ""),
) -> None:
    print("Interactive inference — type 'exit' to stop.")
    while True:
        user_input = input(prompt).strip()
        if user_input.lower() in exit_tokens:
            break
        encoded = encode_fn(user_input)
        prediction = generate_fn(encoded)
        print("\nGenerated article:\n", prediction, "\n")