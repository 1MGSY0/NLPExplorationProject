# NLPExplorationProject

This repository compares Traditional NLP methods vs Neural (transformer-based) methods across multiple datasets and applications. It provides a small UI to test the best-performing model per application.

Structure (initial skeleton)
- `TODO.md` — project checklist and roadmap (claim tasks here)
- `requirements.txt` — Python dependencies
- `src/` — source code stubs: data processing, models, evaluation, UI
- `notebooks/` — EDA and experiment notebooks
- `data/` — raw and processed dataset folders (not included); see `data/README.md` for expected layout
- `models/` — place saved model checkpoints here
- `results/` — evaluation JSON/CSV outputs

Quick start (local dev)
1. Create a Python venv and activate it.
2. Install dependencies:

   pip install -r requirements.txt

3. Start the demo UI (Streamlit):

   streamlit run src/ui/app.py

Notes
- This is a scaffold. Implementations for training and evaluation are in `src/` as stubs and should be filled in per-application.
