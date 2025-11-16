from __future__ import annotations

import json
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gensim.models import Word2Vec
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATASET2_DIR = Path(__file__).resolve().parents[1]
if str(DATASET2_DIR) not in sys.path:
    sys.path.append(str(DATASET2_DIR))

from shared_utils import encode_tokens, segment_sentences, tokenize_sentence  # noqa: E402
from tfidf_retrieval_baseline import load_model_bundle as load_tfidf_bundle  # noqa: E402
from word2vec_similarity_baseline import (  # noqa: E402
    average_embedding,
    cosine_similarity,
    text_tokens,
)
from gru_sentence_classifier import GRUSentenceClassifier, MAX_SEQ_LEN as GRU_MAX_SEQ_LEN  # noqa: E402
from bert_sentence_classifier import BASE_MODEL_NAME, MAX_SEQ_LEN as BERT_MAX_SEQ_LEN  # noqa: E402

logger = logging.getLogger("summarisation-backend")
logging.basicConfig(level=logging.INFO)

MODELS_DIR = DATASET2_DIR / "models"
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_sentence_ranker.pkl"
WORD2VEC_MODEL_PATH = MODELS_DIR / "word2vec_sentence_similarity.model"
GRU_MODEL_PATH = MODELS_DIR / "gru_sentence_classifier.pt"
GRU_VOCAB_PATH = MODELS_DIR / "gru_sentence_vocab.json"
BERT_MODEL_PATH = MODELS_DIR / "bert_sentence_classifier.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SummariseRequest(BaseModel):
    article: str = Field(..., min_length=20, description="Raw article text to summarise.")
    top_k: int = Field(3, ge=1, le=5, description="Number of sentences to return per model.")


class SummariesResponse(BaseModel):
    tfidf: List[str]
    word2vec: List[str]
    gru: List[str]
    bert: List[str]


@lru_cache(maxsize=1)
def get_tfidf_bundle():
    if not TFIDF_MODEL_PATH.exists():
        raise FileNotFoundError(
            "TF-IDF model is missing. Run tfidf_retrieval_baseline.py to train and save the bundle."
        )
    logger.info("Loading TF-IDF bundle from %s", TFIDF_MODEL_PATH)
    return load_tfidf_bundle(str(TFIDF_MODEL_PATH))


@lru_cache(maxsize=1)
def get_word2vec_model() -> Word2Vec:
    if not WORD2VEC_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Word2Vec model is missing. Run word2vec_similarity_baseline.py to train and save the model."
        )
    logger.info("Loading Word2Vec weights from %s", WORD2VEC_MODEL_PATH)
    return Word2Vec.load(str(WORD2VEC_MODEL_PATH))


@lru_cache(maxsize=1)
def get_gru_resources() -> Tuple[GRUSentenceClassifier, Dict[str, int]]:
    if not GRU_MODEL_PATH.exists() or not GRU_VOCAB_PATH.exists():
        raise FileNotFoundError(
            "GRU checkpoint or vocabulary missing. Run gru_sentence_classifier.py to train and save both files."
        )
    logger.info("Loading GRU checkpoint from %s", GRU_MODEL_PATH)
    with open(GRU_VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_payload = json.load(f)
    itos = vocab_payload.get("itos", [])
    stoi = {tok: idx for idx, tok in enumerate(itos)}
    if "<pad>" not in stoi:
        raise RuntimeError("GRU vocabulary missing <pad> token")
    model = GRUSentenceClassifier(len(stoi), stoi["<pad>"])
    state_dict = torch.load(GRU_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model, stoi


@lru_cache(maxsize=1)
def get_bert_resources():
    logger.info("Initialising tokenizer/model for %s", BASE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=1,
        problem_type="single_label_classification",
    )
    if BERT_MODEL_PATH.exists():
        logger.info("Loading fine-tuned BERT weights from %s", BERT_MODEL_PATH)
        state_dict = torch.load(BERT_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        logger.warning("Fine-tuned BERT checkpoint not found; using base backbone weights.")
    model.to(DEVICE).eval()
    return tokenizer, model


def select_top(entries: List[Dict[str, object]], top_k: int, score_key: str) -> List[str]:
    if not entries:
        return []
    ranked = sorted(entries, key=lambda item: (-float(item[score_key]), item["sentence_index"]))
    ordered = sorted(ranked[:top_k], key=lambda item: item["sentence_index"])
    return [item["sentence"] for item in ordered]


def summarise_tfidf(sentences: List[str], top_k: int) -> List[str]:
    bundle = get_tfidf_bundle()
    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]
    X = vectorizer.transform(sentences)
    probs = classifier.predict_proba(X)[:, 1]
    entries = [
        {"sentence": sent, "prob": float(prob), "sentence_index": idx}
        for idx, (sent, prob) in enumerate(zip(sentences, probs))
    ]
    return select_top(entries, top_k, "prob")


def summarise_word2vec(article: str, sentences: List[str], top_k: int) -> List[str]:
    model = get_word2vec_model()
    doc_vec = average_embedding(text_tokens(article), model)
    if doc_vec is None:
        logger.warning("Word2Vec could not build a document vector; returning empty result.")
        return []
    entries = []
    for idx, sentence in enumerate(sentences):
        sent_vec = average_embedding(tokenize_sentence(sentence), model)
        score = cosine_similarity(sent_vec, doc_vec)
        entries.append({"sentence": sentence, "score": score, "sentence_index": idx})
    return select_top(entries, top_k, "score")


def summarise_gru(sentences: List[str], top_k: int) -> List[str]:
    model, stoi = get_gru_resources()
    if not sentences:
        return []
    encoded = []
    lengths = []
    for sentence in sentences:
        tokens = tokenize_sentence(sentence)
        lengths.append(min(len(tokens) + 2, GRU_MAX_SEQ_LEN))
        encoded.append(torch.tensor(encode_tokens(tokens, stoi, GRU_MAX_SEQ_LEN), dtype=torch.long))
    batch = torch.stack(encoded).to(DEVICE)
    length_tensor = torch.tensor(lengths, dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(model(batch, length_tensor)).cpu().tolist()
    entries = [
        {"sentence": sent, "prob": float(prob), "sentence_index": idx}
        for idx, (sent, prob) in enumerate(zip(sentences, probs))
    ]
    return select_top(entries, top_k, "prob")


def summarise_bert(sentences: List[str], top_k: int) -> List[str]:
    tokenizer, model = get_bert_resources()
    if not sentences:
        return []
    encoded = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=BERT_MAX_SEQ_LEN,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
        probs = torch.sigmoid(logits).cpu().tolist()
    entries = [
        {"sentence": sent, "prob": float(prob), "sentence_index": idx}
        for idx, (sent, prob) in enumerate(zip(sentences, probs))
    ]
    return select_top(entries, top_k, "prob")


def summarise_article(article: str, top_k: int) -> Dict[str, List[str]]:
    sentences = segment_sentences(article)
    if not sentences:
        raise ValueError("Unable to detect any sentences in the provided article.")
    results: Dict[str, List[str]] = {}
    errors: Dict[str, str] = {}
    for name, fn in (
        ("tfidf", lambda: summarise_tfidf(sentences, top_k)),
        ("word2vec", lambda: summarise_word2vec(article, sentences, top_k)),
        ("gru", lambda: summarise_gru(sentences, top_k)),
        ("bert", lambda: summarise_bert(sentences, top_k)),
    ):
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s summariser failed", name)
            errors[name] = str(exc)
            results[name] = []
    if all(len(items) == 0 for items in results.values()):
        raise RuntimeError(f"All models failed: {errors}")
    return results


app = FastAPI(title="Dataset2 Summarisation Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/summarise", response_model=SummariesResponse)
async def summarise_endpoint(payload: SummariseRequest):
    try:
        summaries = summarise_article(payload.article, payload.top_k)
        return summaries
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Summarisation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
