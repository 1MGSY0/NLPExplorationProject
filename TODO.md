# NLPExplorationProject - Checklist & Roadmap (Per-dataset plan)

The repository currently stores dataset work under `src/` in these dataset folders:

- `src/Dataset1/`
- `src/Dataset2/`
- `src/Dataset3/`

We've aligned the TODO to reflect the current layout: each dataset folder should contain its own subfolders for raw data, notebooks, models and results. Team members can work inside their dataset folder and keep dataset-specific artifacts colocated.

## Project contract (short)
- Inputs: raw dataset files placed under `src/DatasetX/raw/` (CSV/JSON/Text). If you prefer a central `data/` folder, create it and update this file.
- Outputs: per-dataset best Traditional and Neural models, evaluation reports, saved checkpoints in `src/DatasetX/models/<app>/` (or `models/<dataset>/` if you prefer central storage), and a small dataset-specific UI under `src/ui/` (or `src/DatasetX/ui/`).
- Error modes: missing columns, extreme class imbalance, transformer OOMs on long text. Mitigations: manifest checks, sampling, truncation/striding, small-model configs.

## Per-dataset workflow (apply these steps separately for each dataset folder under `src/`)
1. Place raw files in `src/DatasetX/raw/` and add `src/DatasetX/manifest.md` describing files and expected columns and any preprocessing notes.
2. Create an EDA notebook `src/DatasetX/notebooks/<dataset>-eda.ipynb` documenting schema, label distributions, and sample rows. (Notebooks can be placed at project-level `notebooks/` too — the repo currently uses per-dataset subfolders.)
3. Implement dataset-specific preprocessing in `src/DatasetX/preprocessing.py` or in the common `src/data_processing.py` if logic is small. Include cleaning, tokenization, label creation (e.g., NLTK VADER for weak sentiment labels), and filtering rules (e.g., remove summaries shorter than N words).
4. Implement Traditional baseline(s) for the dataset. Place model code under `src/DatasetX/models/traditional.py` or use the common `src/models/traditional.py` and dataset-specific notebooks in `src/DatasetX/notebooks/`.
5. Implement Neural baseline(s) under `src/DatasetX/models/` or in `src/models/nn_models.py` with dataset-specific training scripts in `src/DatasetX/`.
6. Train, validate and save models to `src/DatasetX/models/<app>/` and store evaluation metrics in `src/DatasetX/results/<dataset>-<app>-results.json`.
7. Create a dataset-specific demo UI `src/ui/<dataset>_app.py` or `src/DatasetX/ui/app.py` that loads the saved best Traditional and Neural models and provides side-by-side inference for manual testing. (Either location is fine; record the canonical path in `src/DatasetX/manifest.md`.)
8. Write a short report `src/DatasetX/reports/<dataset>-summary.md` describing dataset quirks, preprocessing, model results and recommended best model.

## Datasets & Applications (per-dataset tasks)

### Amazon Book Reviews
Source: https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews

- App A: Sentiment analysis
  - Traditional: Naive Bayes (TF-IDF + MultinomialNB)
  - Neural: BERT/DistilBERT fine-tune
  - Metrics: Precision, Recall, Accuracy, F1, Macro-F1

- App B: Summarization of reviews
  - Traditional (extractive): TextRank (gensim) or sentence scoring
  - Neural (abstractive): T5 (t5-small / t5-base)
  - Metrics: BLEU, ROUGE-1/2/L, METEOR

### CNN/DailyMail
Source: https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail

- App A: Generate article from highlights (seq2seq)
  - Neural: T5/BART fine-tune
  - Traditional baseline: extractive/retrieval heuristic (optional)
  - Metrics: ROUGE, BLEU, METEOR

- App B: News categorisation
  - Traditional: TF-IDF + LinearSVC (SVM)
  - Neural: RoBERTa/BERT fine-tune
  - Metrics: Accuracy, Precision, Recall, F1

### Emotion Dataset
Source: https://www.kaggle.com/datasets/parulpandey/emotion-dataset

- App A: Emotion classification
  - Traditional: TF-IDF + LogisticRegression/SVM
  - Neural: BERT/RoBERTa fine-tune; optional LLM (ChatGPT) comparisons
  - Metrics: Accuracy, Precision, Recall, F1, Macro-F1

- App B: (optional extension) Emotion intensity / span detection
  - Traditional: rule-based + regression/classifier
  - Neural: sequence labeling or regressor with transformers
  - Metrics: task-dependent (e.g., Pearson/Spearman for intensity)

## UI approach
- Each dataset gets a dataset-specific UI under `src/ui/` (e.g., `src/ui/amazon_app.py`, `src/ui/cnn_app.py`, `src/ui/emotion_app.py`). These apps load the saved models for that dataset and provide side-by-side comparisons between the chosen Traditional and Neural models.
- Optionally, create an aggregator UI `src/ui/main_app.py` that can launch dataset-specific demos or show an overview dashboard.

## Ownership suggestions (update with names)
- Amazon: Sentiment — (assign member)
- Amazon: Summarization — (assign member)
- CNN/DailyMail: Generation — (assign member)
- CNN/DailyMail: Categorisation — (assign member)
- Emotion: Classification — (assign member)
- Emotion: Extension — (assign member)
- UI (per-dataset) — each dataset owner can create their dataset UI; a central UI integrator can be assigned to `main_app.py`.
- Experiments & reporting — collate `results/` and produce final comparative report.


## Acceptance criteria & quality gates
- Per-app deliverables: preprocessing script/notebook, training/inference script(s), evaluation JSON, saved best-model artifacts, short notes.
- Reproducibility: include seed, model/tokenizer names, and training args in `results/` metadata.
- Tests: unit tests for preprocessing functions and evaluation helpers; smoke test for a minimal end-to-end run.

## Next actions (pick one and claim)
1. Create `data/<dataset>/manifest.md` for the dataset you claim and add raw files to `data/<dataset>/raw/` (do not commit raw data to Git).
2. Implement the EDA notebook `notebooks/<dataset>-eda.ipynb` and push observations.
3. Implement dataset-specific preprocessing function(s) in `src/data_processing.py` and add a unit test.

---
Last updated: 2025-11-07
