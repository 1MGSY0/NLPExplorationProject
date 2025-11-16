from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
)

STAGE_NAME = "tfidf_sentence_ranker"
MODEL_PATH = "./models/tfidf_sentence_ranker.pkl"
METRICS_PATH = "./results/tfidf_sentence_ranker_metrics.json"
SAMPLES_PATH = "./results/tfidf_sentence_ranker_samples.json"

# Fixed configuration (no CLI flags required)
TRAIN_ARTICLES = 20_000
VAL_ARTICLES = 2_000
TEST_ARTICLES = 2_000
TOP_K_SENTENCES = 3
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_MAX = 2
TFIDF_MIN_DF = 2
ROUGE_LABEL_THRESHOLD = 0.35
DECISION_THRESHOLD = 0.5
LOGREG_MAX_ITER = 1000
RUN_INTERACTIVE = False
GLOBAL_SEED = DEFAULT_SEED

STAGE_CONFIG = {
    "train_articles": TRAIN_ARTICLES,
    "val_articles": VAL_ARTICLES,
    "test_articles": TEST_ARTICLES,
    "top_k_sentences": TOP_K_SENTENCES,
    "tfidf_max_features": TFIDF_MAX_FEATURES,
    "tfidf_ngram_max": TFIDF_NGRAM_MAX,
    "tfidf_min_df": TFIDF_MIN_DF,
    "rouge_label_threshold": ROUGE_LABEL_THRESHOLD,
    "decision_threshold": DECISION_THRESHOLD,
    "logreg_max_iter": LOGREG_MAX_ITER,
    "seed": GLOBAL_SEED,
    "data_dir": DEFAULT_DATA_DIR,
    "train_file": DEFAULT_TRAIN_FILE,
    "val_file": DEFAULT_VAL_FILE,
    "test_file": DEFAULT_TEST_FILE,
    "run_interactive": RUN_INTERACTIVE,
}


def build_vectorizer(max_features: int, ngram_max: int, min_df: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, max(1, ngram_max)),
        min_df=min_df,
        lowercase=True,
        norm="l2",
    )


def load_split_data(
    split: str,
    n_articles: int | None,
    rouge_threshold: float,
    seed: int = GLOBAL_SEED,
):
    sentence_df, article_records = build_sentence_dataset(
        split,
        n_articles=n_articles,
        rouge_threshold=rouge_threshold,
        data_dir=DEFAULT_DATA_DIR,
        train_file=DEFAULT_TRAIN_FILE,
        val_file=DEFAULT_VAL_FILE,
        test_file=DEFAULT_TEST_FILE,
        seed=seed,
    )
    return sentence_df, article_records


def train_classifier(
    vectorizer: TfidfVectorizer,
    sentences: List[str],
    labels: List[int],
    max_iter: int,
) -> Tuple[TfidfVectorizer, LogisticRegression]:
    X_train = vectorizer.fit_transform(sentences)
    classifier = LogisticRegression(max_iter=max_iter, class_weight="balanced", solver="liblinear")
    classifier.fit(X_train, labels)
    return vectorizer, classifier


def _summarise_article(entries: List[Dict[str, object]], top_k: int) -> Tuple[List[int], str]:
    if not entries:
        return [], ""
    ranked = sorted(entries, key=lambda item: (-item["prob"], item["sentence_index"]))
    chosen = ranked[:top_k]
    ordered = sorted(chosen, key=lambda item: item["sentence_index"])
    summary = " ".join(item["sentence"] for item in ordered)
    return [item["sentence_index"] for item in ordered], summary


def evaluate_split(
    name: str,
    sentence_df,
    articles,
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    top_k: int,
    decision_threshold: float,
    sample_cap: int = 10,
):
    if sentence_df.empty:
        raise RuntimeError(f"No sentence samples available for {name} split")
    X = vectorizer.transform(sentence_df["sentence"].tolist())
    probs = classifier.predict_proba(X)[:, 1]
    preds = (probs >= decision_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        sentence_df["label"].astype(int), preds, average="binary", zero_division=0
    )

    grouped = defaultdict(list)
    for row, prob in zip(sentence_df.itertuples(), probs):
        grouped[row.article_id].append(
            {
                "sentence": row.sentence,
                "label": int(row.label),
                "prob": float(prob),
                "sentence_index": int(row.sentence_index),
            }
        )

    summaries: List[str] = []
    references: List[str] = []
    samples: List[Dict[str, str]] = []
    for record in articles:
        entries = grouped.get(record["article_id"], [])
        _, summary_text = _summarise_article(entries, top_k)
        summaries.append(summary_text)
        references.append(record["highlights"])
        if len(samples) < sample_cap:
            samples.append(
                {
                    "article_id": record["article_id"],
                    "prediction": summary_text,
                    "reference": record["highlights"],
                }
            )

    rouge_metrics = compute_all_metrics(summaries, references)
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        **rouge_metrics,
    }
    print(
        f"{name}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
        f"| ROUGE-1={metrics['rouge1']:.2f} ROUGE-L={metrics['rougeL']:.2f}"
    )
    return metrics, samples


def save_model_bundle(path: str, vectorizer: TfidfVectorizer, classifier: LogisticRegression, top_k: int) -> None:
    bundle = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "top_k": top_k,
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_model_bundle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def interactive_cli(bundle) -> None:
    vectorizer: TfidfVectorizer = bundle["vectorizer"]
    classifier: LogisticRegression = bundle["classifier"]
    top_k: int = bundle.get("top_k", 3)
    print("Interactive TF-IDF extractor — paste an article (blank line to quit).")
    while True:
        user_input = input("Article> ").strip()
        if user_input.lower() in {"", "exit", "quit"}:
            break
        sentences = segment_sentences(user_input)
        if not sentences:
            print("No sentences detected.")
            continue
        X = vectorizer.transform(sentences)
        probs = classifier.predict_proba(X)[:, 1]
        entries = [
            {"sentence": sent, "prob": float(prob), "sentence_index": idx}
            for idx, (sent, prob) in enumerate(zip(sentences, probs))
        ]
        _, summary = _summarise_article(entries, top_k)
        print("\nTop sentences:\n", summary or "<none>", "\n")


def main() -> None:
    set_global_seed(GLOBAL_SEED)
    ensure_dir("./models")
    ensure_dir("./results")
    ensure_dir(os.path.dirname(MODEL_PATH) or ".")

    start_time = time.time()
    train_df, train_articles = load_split_data(
        "train",
        n_articles=TRAIN_ARTICLES,
        rouge_threshold=ROUGE_LABEL_THRESHOLD,
    )
    val_df, val_articles = load_split_data(
        "val",
        n_articles=VAL_ARTICLES,
        rouge_threshold=ROUGE_LABEL_THRESHOLD,
    )
    test_df, test_articles = load_split_data(
        "test",
        n_articles=TEST_ARTICLES,
        rouge_threshold=ROUGE_LABEL_THRESHOLD,
    )

    vectorizer = build_vectorizer(TFIDF_MAX_FEATURES, TFIDF_NGRAM_MAX, TFIDF_MIN_DF)
    vectorizer, classifier = train_classifier(
        vectorizer,
        train_df["sentence"].tolist(),
        train_df["label"].astype(int).tolist(),
        max_iter=LOGREG_MAX_ITER,
    )

    val_metrics, _ = evaluate_split(
        "Validation",
        val_df,
        val_articles,
        vectorizer,
        classifier,
        TOP_K_SENTENCES,
        DECISION_THRESHOLD,
        sample_cap=5,
    )
    test_metrics, samples = evaluate_split(
        "Test",
        test_df,
        test_articles,
        vectorizer,
        classifier,
        TOP_K_SENTENCES,
        DECISION_THRESHOLD,
        sample_cap=10,
    )

    elapsed = time.time() - start_time
    train_history = [{"epoch": 1, "loss": 0.0}]
    val_history = [{**val_metrics, "epoch": 1}]

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"validation": val_metrics, "test": test_metrics}, f, indent=2)
    with open(SAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    save_model_bundle(MODEL_PATH, vectorizer, classifier, TOP_K_SENTENCES)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    report_path = save_report(
        STAGE_NAME,
        train_history,
        val_history,
        test_metrics,
        samples,
        training_time=elapsed,
        config=STAGE_CONFIG,
        output_dir="./results",
        run_id=run_id,
    )
    print(f"Saved TF-IDF sentence ranker to {MODEL_PATH} and report to {report_path}.")

    if RUN_INTERACTIVE:
        bundle = load_model_bundle(MODEL_PATH)
        interactive_cli(bundle)


if __name__ == "__main__":
    main()
