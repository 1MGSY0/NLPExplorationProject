# Summarisation Dashboard

A lightweight React (Vite + TypeScript) interface that showcases evaluation metrics and a live demo for extractive summarisation models.

## Features

- **Evaluation Dashboard** (`/`): loads `reports.json` and displays both a metrics table and an interactive Recharts bar chart.
- **Summarisation Demo** (`/demo`): send custom articles to `POST http://localhost:8000/summarise` and compare model outputs side-by-side.
- Minimal responsive styling with a shared navbar linking both pages.

## Getting Started

```bash
cd src/dashboard-app
npm install
npm run dev
```

Visit `http://localhost:5173` in your browser. Ensure the summarisation backend is available at `http://localhost:8000/summarise` for the demo page.

## Build for Production

```bash
cd src/dashboard-app
npm run build
npm run preview
```

This runs the Vite build and serves the static assets locally so you can verify the bundle.
