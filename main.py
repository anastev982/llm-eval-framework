import argparse
from datetime import datetime
from eval.loaders import load_jsonl
from eval.metrics import accuracy, rouge1, token_f1
from eval.lim_client import call_llm
from utils.helpers import normalize_span, save_errors_csv, print_llm_summary, open_report
from utils.reporting import generate_html_report
import os, webbrowser
import json

# CLASSIFICATION TASK

def evaluate_classification(path: str, model: str):
    examples = list(load_jsonl(path))
    preds = []
    golds = []

    for ex in examples:
        text = ex["input"]
        label = ex["expected_label"].strip().lower()

        system_prompt = (
            "You are a text classifier. You MUST answer with exactly one of the "
            "following labels: finance, science, sports, other.\n"
            "Answer with the label only. No explanation."
        )
        user_prompt = f"Text: {text}\nLabel:"

        pred = call_llm(system_prompt, user_prompt, model=model)
        pred = pred.strip().lower()

        print("INPUT:", text)
        print("PRED:", pred)
        print("GOLD:", label)
        print("-" * 40)

        preds.append(pred)
        golds.append(label)

    acc = accuracy(preds, golds)
    print(f"[CLASSIFICATION] Accuracy: {acc:.3f}")
    return acc


# SUMMARIZATION TASK

def evaluate_summarization(path: str, model: str):
    print("\nRunning summarization evaluation...\n")

    scores = []
    errors = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            text = example["input"]
            gold_summary = example["reference_summary"]

            system_prompt = (
                "You are a summarization assistant. "
                "Your goal is to maximize ROUGE-1 F1 overlap with a reference summary.\n\n"
                "Rules:\n"
                "- Use key phrases from the article.\n"
                "- Preserve important nouns and named entities.\n"
                "- Avoid synonyms unless necessary.\n"
                "- Keep meaning faithful.\n"
                "- Write 2–3 sentences.\n"
            )
            user_prompt = f"Article:\n\n{text}\n\nSummary:"

            pred = call_llm(system_prompt, user_prompt, model=model)

            _, _, f1 = rouge1(pred, gold_summary)
            scores.append(f1)

            if f1 < 0.5:
                errors.append({
                    "input": text,
                    "pred": pred,
                    "gold": gold_summary,
                    "f1": round(f1, 3)
                })

            short_input = text[:120] + "..." if len(text) > 120 else text
            print("INPUT:", short_input)
            print("PRED:", pred)
            print("GOLD:", gold_summary)
            print("-" * 40)

    avg_f1 = sum(scores) / len(scores)
    print(f"[SUMMARIZATION] ROUGE-1 F1: {avg_f1:.3f}")
    print("\n[ERROR ANALYSIS] Low-scoring summarization examples:")

    if not errors:
        print("  No low-score examples ")
    else:
        for err in errors:
            print("-" * 40)
            print(f"F1: {err['f1']}")
            print(f"INPUT: {err['input'][:120]}...")
            print(f"PRED: {err['pred']}")
            print(f"GOLD: {err['gold']}")
            print()

        save_errors_csv(errors, model, task="summarization")

    return avg_f1


# EXTRACTION TASK

def evaluate_extraction(path: str, model: str, output_dir: str = "logs"):
    f1_scores = []
    errors = []

    for ex in load_jsonl(path):
        text = ex["input"]
        ref = ex["reference"]

        system_prompt = (
            "You extract ONLY the time expression that describes work or study "
            "experience from the text. Return ONLY that expression."
        )

        try:
            pred = call_llm(system_prompt, text, model=model)
        except Exception:
            pred = "<ERROR>"

        norm_pred = normalize_span(pred)
        norm_ref = normalize_span(ref)

        f1 = token_f1(norm_pred, norm_ref)
        f1_scores.append(f1)

        if f1 < 1.0:
            errors.append({
                "input": text,
                "pred": pred,
                "normalized_pred": norm_pred,
                "gold": ref,
                "normalized_gold": norm_ref,
                "f1": f1
            })

    avg_f1 = sum(f1_scores) / len(f1_scores)
    print(f"[EXTRACTION] Avg token-F1: {avg_f1:.3f}")

    save_errors_csv(
        errors, 
        model, 
        task="extraction",
        output_dir=output_dir
        )
    
    return avg_f1

# MAIN


def main():
    parser = argparse.ArgumentParser(description="LLM Evaluation Framework")

    parser.add_argument("--task",
                        choices=["all", "classification", "summarization", "extraction"],
                        default="all")

    parser.add_argument("--model", default="gpt-4o-mini")
    
    parser.add_argument("--experiment-id",
                        default=None,
                        help="Experiment identifier (e.g. summarization_v2_promptA)")

    parser.add_argument("--classification-file",
                        default="data/classification_news.jsonl")

    parser.add_argument("--summarization-file",
                        default="data/summarization_articles.jsonl")

    parser.add_argument("--extraction-file",
                        default="data/extraction_experience.jsonl")

    args = parser.parse_args()
    
    experiment_id = (
        args.experiment_id if args.experiment_id is not None
        else f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    base_exp_dir = os.path.join("logs", "experiments", experiment_id)
    os.makedirs(base_exp_dir, exist_ok=True)

    print("\nRunning LLM Evaluation Framework...\n")

    TASKS = {
        "classification": (evaluate_classification, args.classification_file),
        "summarization": (evaluate_summarization, args.summarization_file),
        "extraction": (lambda path, model: evaluate_extraction(path, model, base_exp_dir), args.extraction_file),
    }

    model_names = [m.strip() for m in args.model.split(",")]
    results = {}

    for model_name in model_names:
        model_results = {}

        if args.task == "all":
            for name, (fn, path) in TASKS.items():
                model_results[name] = fn(path, model_name)
        else:
            name = args.task
            fn, path = TASKS[name]
            model_results[name] = fn(path, model_name)

        results[model_name] = model_results

    print_llm_summary(results)
    report_path = generate_html_report(
        results, 
        output_dir=base_exp_dir
        )
    
    with open(os.path.join(base_exp_dir, "experiment.json"), "w") as f:
        json.dump(
            {
                "experiment_id": experiment_id,
                "models": model_names,
                "task": args.task,
                "timestamp": datetime.now().isoformat(),
                "results": results,
            },
            f,
            indent=2,
        )
    
    try:
        abs_path = os.path.abspath(report_path)
        webbrowser.open(f"file://{abs_path}")
    except Exception as e:
        print(f"[REPORT] Could not open browser automatically: {e}")


if __name__ == "__main__":
    main()
