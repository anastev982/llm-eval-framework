from datetime import datetime
import os
from html import escape
from typing import Dict, Any

from utils.scoring import avg_score

def generate_html_report(
    results: Dict[str, Dict[str, Any]],
    output_dir: str = "logs/report",
) -> str:
    """Minimal HTML report + Chart.js bar graf za results by model/task."""
    if not results:
        raise ValueError("No results to report.")

    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, "report.html")

    # 1) Primary lists: models i tasks
    model_names = sorted(
        results.keys(),
        key=lambda m: avg_score(results[m]),
        reverse=True
        )
    task_names = list(next(iter(results.values())).keys())

    # 2) HTML table (rows) 
    rows = ""
    for model in model_names:
        scores = results[model]
        cells = "".join(
            f"<td>{escape(f'{scores.get(t):.3f}' if isinstance(scores.get(t), (int, float)) else 'N/A')}</td>"
            for t in task_names
        )
        avg = avg_score(scores)
        rows += (
            f"<tr>"
            f"<td>{escape(model)}</td>"
            f"{cells}"
            f"<td><b>{avg:.3f}</b></td>"
            f"</tr>"
        )

    # --- 3) Chart.js data (chart_data -> js_datasets) ---
    chart_data = {task: [] for task in task_names}
    for task in task_names:
        for m in model_names:
            val = results[m].get(task)
            chart_data[task].append(float(val) if isinstance(val, (int, float)) else 0.0)

    js_models = "[" + ", ".join(f"'{m}'" for m in model_names) + "]"

    js_datasets_parts = []
    for task in task_names:
        data_list = chart_data[task]
        js_data = "[" + ", ".join(f"{v:.3f}" for v in data_list) + "]"
        js_datasets_parts.append(
            "{label: '" + task.capitalize() + "', data: " + js_data + "}"
        )
    js_datasets = "[" + ", ".join(js_datasets_parts) + "]"

    # --- 4) HTML + Chart.js ---
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>LLM Evaluation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; margin-bottom: 2rem; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 12px; }}
    th {{ background: #eee; }}
  </style>
  <!-- CDN za Chart.js -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <h1>LLM Evaluation Report</h1>
  <p>Generated at {datetime.now()}</p>

  <h2>Summary</h2>
  <table>
    <tr>
      <th>Model</th>
      {''.join(f'<th>{escape(t.capitalize())}</th>' for t in task_names)}
      <th>Avg</th>
    </tr>
    {rows}
  </table>

  <h2>Summary (chart)</h2>
  <canvas id="scoresChart"></canvas>

  <h2>Error Logs</h2>
  <p>CSV files are saved in the <code>logs/</code>:</p>
  <ul>
    <li>Summarization: <code>logs/summarization_errors_&lt;model&gt;.csv</code></li>
    <li>Extraction: <code>logs/extraction_errors_&lt;model&gt;.csv</code></li>
  </ul>

  <script>
    const models = {js_models};
    const datasets = {js_datasets};

    const ctx = document.getElementById('scoresChart').getContext('2d');
    new Chart(ctx, {{
      type: 'bar',
      data: {{
        labels: models,
        datasets: datasets
      }},
      options: {{
        responsive: true,
        scales: {{
          y: {{
            beginAtZero: true,
            max: 1.0
          }}
        }}
      }}
    }});
  </script>
</body>
</html>
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[REPORT] Saved HTML report → {out_path}")
    return out_path
