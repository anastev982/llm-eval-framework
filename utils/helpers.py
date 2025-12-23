from html import escape
import re
import string
import os
import subprocess
from typing import Dict, Any
import webbrowser

def normalize_span(s: str) -> str:
    if not isinstance(s, str):
        return ""

    s = s.lower().replace("-", " ")

    s = re.sub(
        r"\b(internship|experience|role|position|overall|"
        r"more than|over|at least|about|around|approximately|almost)\b",
        " ",
        s,
    )

    s = re.sub(r"\byears?\b", "year", s)
    s = re.sub(r"\bmonths?\b", "month", s)

    s = re.sub(r"\bof\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    m = re.search(r"\d+\s+(year|month)", s)
    if m:
        return m.group(0)

    m = re.search(r"\b\d{4}\b", s)
    if m:
        return m.group(0)

    return s

def normalize_text(s: str) -> str:
    """Lowercase + remove punctuation + normalize spaces."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def save_errors_csv(errors, model, task, output_dir="logs"):
    if not errors:
        return

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{task}_errors_{model}.csv")

    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=errors[0].keys())
        writer.writeheader()
        writer.writerows(errors)

    print(f"[ERROR LOG] Saved {len(errors)} examples {path}")
    
def print_llm_summary(results: dict):
    if not results:
        print("No results to summarize.")
        return

    print("------- LLM EVALUATION SUMMARY -------")

    names = list(next(iter(results.values())).keys())

    # header
    print("Model", *[t.capitalize() for t in names], sep="\t")
    
    for model_name, scores in results.items():
        row = [model_name] + [
            f"{scores.get(t):.3f}" if isinstance(scores.get(t), (int, float)) else "N/A"
            for t in names
        ]
        print(*row, sep="\t")

def open_report(path: str):
    abs_path = os.path.abspath(path)

    # WSL detection (usually sufficient)
    try:
        is_wsl = "microsoft" in os.uname().release.lower()  # type: ignore
    except AttributeError:
        # os.uname() not available on Windows
        is_wsl = False

    if is_wsl:
        # Most stable in WSL: open through Windows shell
        subprocess.run(["explorer.exe", abs_path], check=False)
    else:
        webbrowser.open(f"file://{abs_path}")

def summarization_cell(model, score):
    csv_path = f"logs/summarization_errors_{model}.csv"
    return (
        f"{score:.3f} "
        f"<a href='../{csv_path}'>errors</a>"
        if isinstance(score, (int, float))
        else "N/A"
    )
