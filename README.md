Research-focused evaluation framework.  
> This project is intended for experimentation and benchmarking, not production deployment.

# LLM Evaluation Framework

An end-to-end LLM evaluation framework for benchmarking multiple large language models
across **classification**, **summarization**, and **information extraction** tasks,
with automated metrics, error analysis, and HTML reports.

---

## Overview

This project provides a lightweight, reproducible framework for evaluating and
comparing LLM performance on common NLP tasks using reference-based metrics.

It is designed to support:

- prompt experimentation
- multi-model comparison
- systematic error analysis

---

## Supported Tasks & Metrics

| Task           | Description                 | Metric         |
| -------------- | --------------------------- | -------------- |
| Classification | News topic classification   | Accuracy       |
| Summarization  | Short abstractive summaries | ROUGE-1 F1     |
| Extraction     | Time span extraction        | Token-level F1 |

---

## Key Features

- Multi-model evaluation in a single run
- Reference-based automatic metrics
- CSV error logs for qualitative analysis
- Interactive HTML report with tables and charts
- Clean, extensible project structure

---

## Project Structure

```text
llm-eval-framework/
├── main.py                 # Entry point
├── eval/
│   ├── loaders.py          # JSONL data loading
│   ├── metrics.py          # Accuracy, ROUGE-1, token-F1
│   ├── lim_client.py       # OpenAI client wrapper
│   └── judge.py            # Experimental LLM-as-a-judge (not integrated)
├── utils/
│   ├── helpers.py          # Text normalization, logging
│   ├── scoring.py          # Score aggregation
│   └── reporting.py        # HTML report generation
├── data/
│   ├── classification_news.jsonl
│   ├── summarization_articles.jsonl
│   └── extraction_experience.jsonl
└── logs/
    └── experiments/        # Generated reports and logs

Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt```

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your_api_key_here"```

Usage

Run all tasks on one or more models:

```bash
python main.py --model gpt-4o-mini,gpt-4o```

Run a single task:

```bash
python main.py --task summarization --model gpt-4o```

Output
Console summary of scores per model and task
CSV files with low-scoring examples for error analysis
HTML report with tables and visualizations saved under:

```bash
logs/experiments/<experiment_id>/report.html```

Notes & Limitations
ROUGE-1 measures lexical overlap and does not fully capture semantic similarity.

The judge.py module contains an experimental LLM-as-a-judge approach,
intended for future integration.
```


