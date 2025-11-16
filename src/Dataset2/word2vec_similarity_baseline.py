from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import precision_recall_fscore_support

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
    build_sentence_dataset,
    compute_all_metrics,
    ensure_dir,
    save_report,
    segment_sentences,
    set_global_seed,
    tokenize_sentence,
)

STAGE_NAME = "word2vec_sentence_similarity"
MODEL_PATH = "./models/word2vec_sentence_similarity.model"
METRICS_PATH = "./results/word2vec_sentence_similarity_metrics.json"
SAMPLES_PATH = "./results/word2vec_sentence_similarity_samples.json"

TRAIN_ARTICLES = 20_000
VAL_ARTICLES = 2_000
TEST_ARTICLES = 2_000
ROUGE_LABEL_THRESHOLD = 0.35
TOP_K_SENTENCES = 3
SIMILARITY_THRESHOLD = 0.35
VECTOR_SIZE = 200
WINDOW_SIZE = 5
MIN_COUNT = 3
NEGATIVE_SAMPLES = 10
EPOCHS = 5
WORKERS = max(1, os.cpu_count() or 1)
SG = 1  # Skip-gram produces better semantics than CBOW for similarity

DEV_MODE = os.getenv("DATASET2_FAST_DEV_RUN", "0").lower() in {"1", "true", "yes"}
if DEV_MODE:
    TRAIN_ARTICLES = min(TRAIN_ARTICLES, 2_000)
    VAL_ARTICLES = min(VAL_ARTICLES, 400)
    TEST_ARTICLES = min(TEST_ARTICLES, 400)

GLOBAL_SEED = DEFAULT_SEED
STAGE_CONFIG: Dict[str, object] = {
    "train_articles": TRAIN_ARTICLES,
    "val_articles": VAL_ARTICLES,
    "test_articles": TEST_ARTICLES,
    "top_k_sentences": TOP_K_SENTENCES,
    "similarity_threshold": SIMILARITY_THRESHOLD,
    "vector_size": VECTOR_SIZE,
    "window_size": WINDOW_SIZE,
    "min_count": MIN_COUNT,
    "negative_samples": NEGATIVE_SAMPLES,
    "epochs": EPOCHS,
    "workers": WORKERS,
    "sg": SG,
    "seed": GLOBAL_SEED,
    "dev_mode": DEV_MODE,
    "data_dir": DEFAULT_DATA_DIR,
    "train_file": DEFAULT_TRAIN_FILE,
    "val_file": DEFAULT_VAL_FILE,
    "test_file": DEFAULT_TEST_FILE,
}


def build_training_corpus(article_records: List[Dict[str, object]]) -> List[List[str]]:
    corpus: List[List[str]] = []
    for record in article_records:
        for sentence in record.get("sentences", []):
            tokens = tokenize_sentence(sentence)
            if tokens:
                corpus.append(tokens)
        highlight_sentences = segment_sentences(record.get("highlights", ""))
        for highlight_sentence in highlight_sentences:
            tokens = tokenize_sentence(highlight_sentence)
            if tokens:
                corpus.append(tokens)
    return corpus


def train_word2vec(corpus: List[List[str]]) -> Word2Vec:
    if not corpus:
        raise RuntimeError("Empty corpus provided to Word2Vec trainer")
    model = Word2Vec(
        sentences=corpus,
        vector_size=VECTOR_SIZE,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        negative=NEGATIVE_SAMPLES,
        sg=SG,
        epochs=EPOCHS,
        workers=WORKERS,
        seed=GLOBAL_SEED,
    )
    return model


def ensure_model(corpus: List[List[str]]) -> Word2Vec:
    if os.path.exists(MODEL_PATH):
        return Word2Vec.load(MODEL_PATH)
    ensure_dir(os.path.dirname(MODEL_PATH) or ".")
    model = train_word2vec(corpus)
    model.save(MODEL_PATH)
    return model


def average_embedding(tokens: Iterable[str], model: Word2Vec) -> Optional[np.ndarray]:
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def cosine_similarity(vec_a: Optional[np.ndarray], vec_b: Optional[np.ndarray]) -> float:
    if vec_a is None or vec_b is None:
        return 0.0
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def text_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    for sentence in segment_sentences(text):
        tokens.extend(tokenize_sentence(sentence))
    return tokens


def attach_similarity_scores(sentence_df, article_records, model: Word2Vec):
    article_map = {record["article_id"]: record for record in article_records}
    highlight_cache: Dict[str, Optional[np.ndarray]] = {}
    scores: List[float] = []
    for row in sentence_df.itertuples():
        if row.article_id not in article_map:
            raise KeyError(f"Missing article metadata for article_id={row.article_id}")
        record = article_map[row.article_id]
        if row.article_id not in highlight_cache:
            highlight_cache[row.article_id] = average_embedding(text_tokens(record.get("highlights", "")), model)
        highlight_vec = highlight_cache[row.article_id]
        sentence_vec = average_embedding(tokenize_sentence(row.sentence), model)
        scores.append(cosine_similarity(sentence_vec, highlight_vec))
    sentence_df = sentence_df.copy()
    sentence_df["score"] = scores
    return sentence_df


def _summarise(entries: List[Dict[str, object]], top_k: int) -> Tuple[List[int], str]:
    if not entries:
        return [], ""
    ranked = sorted(entries, key=lambda item: (-item["score"], item["sentence_index"]))
    chosen = ranked[:top_k]
    ordered = sorted(chosen, key=lambda item: item["sentence_index"])
    summary = " ".join(item["sentence"] for item in ordered)
    indices = [item["sentence_index"] for item in ordered]
    return indices, summary


def evaluate_split(
    name: str,
    sentence_df,
    article_records,
    model: Word2Vec,
    top_k: int,
    threshold: float,
    sample_cap: int = 10,
):
    if sentence_df.empty:
        raise RuntimeError(f"No data available for {name} split")
    scored_df = attach_similarity_scores(sentence_df, article_records, model)
    preds = (scored_df["score"] >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        scored_df["label"].astype(int), preds, average="binary", zero_division=0
    )

    grouped = defaultdict(list)
    for row in scored_df.itertuples():
        grouped[row.article_id].append(
            {
                "sentence": row.sentence,
                "score": row.score,
                "sentence_index": row.sentence_index,
            }
        )

    article_map = {record["article_id"]: record for record in article_records}
    summaries: List[str] = []
    references: List[str] = []
    samples: List[Dict[str, object]] = []

    for article_id, entries in grouped.items():
        idxs, summary = _summarise(entries, top_k)
        highlight_text = article_map[article_id]["highlights"]
        summaries.append(summary)
        references.append(highlight_text)
        if len(samples) < sample_cap:
            samples.append(
                {
                    "article_id": article_id,
                    "highlights": highlight_text,
                    "predicted_summary": summary,
                    "top_sentence_indices": idxs,
                    "scores": [entry["score"] for entry in entries],
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


def interactive_cli(model: Word2Vec, top_k: int):
    print("Interactive Word2Vec similarity — type 'exit' to quit.")
    while True:
        article = input("Article> ").strip()
        if article.lower() in {"", "exit", "quit"}:
            break
        ref = input("Optional reference summary (Enter to skip)> ").strip()
        highlight_text = ref if ref else article
        sentences = segment_sentences(article)
        if not sentences:
            print("No sentences detected.\n")
            continue
        highlight_vec = average_embedding(text_tokens(highlight_text), model)
        entries = []
        for idx, sentence in enumerate(sentences):
            sentence_vec = average_embedding(tokenize_sentence(sentence), model)
            score = cosine_similarity(sentence_vec, highlight_vec)
            entries.append({"sentence": sentence, "score": score, "sentence_index": idx})
        idxs, summary = _summarise(entries, top_k)
        print("\nTop sentences:\n", summary or "<none>", "\n")
        print("Sentence indices:", idxs)


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

    corpus = build_training_corpus(train_articles)
    model = ensure_model(corpus)

    train_metrics, _ = evaluate_split(
        "Train",
        train_df,
        train_articles,
        model,
        TOP_K_SENTENCES,
        SIMILARITY_THRESHOLD,
    )
    val_metrics, val_samples = evaluate_split(
        "Validation",
        val_df,
        val_articles,
        model,
        TOP_K_SENTENCES,
        SIMILARITY_THRESHOLD,
    )
    test_metrics, test_samples = evaluate_split(
        "Test",
        test_df,
        test_articles,
        model,
        TOP_K_SENTENCES,
        SIMILARITY_THRESHOLD,
    )

    elapsed = time.time() - start_time
    train_history = [
        {
            "phase": "word2vec_training",
            "epochs": EPOCHS,
            "vector_size": VECTOR_SIZE,
            "window": WINDOW_SIZE,
            "min_count": MIN_COUNT,
            "negative": NEGATIVE_SAMPLES,
        }
    ]
    val_history = [{**val_metrics, "epoch": 1}]

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
    print(f"Saved Word2Vec similarity model to {MODEL_PATH} and report to {report_path}.")

    if os.getenv("WORD2VEC_INTERACTIVE", "0").lower() in {"1", "true", "yes"}:
        interactive_cli(model, TOP_K_SENTENCES)


if __name__ == "__main__":
    main()
