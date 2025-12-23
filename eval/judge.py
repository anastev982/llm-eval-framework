import json
from eval.lim_client import call_llm

SYSTEM_PROMPT = """
You are an impartial evaluator of AI model outputs.
Return only a JSON object with:
- score (1 to 5)
- justification (brief reasoning)
"""

JUDGE_TEMPLATE = """
Task: {task_type}

Input:
{input}

Model Answer:
{model_answer}

Reference Answer:
{reference}

Evaluate the model answer and return JSON only.
"""

def judge_example(task_type, input_text, model_answer, reference):
    prompt = JUDGE_TEMPLATE.format(
        task_type=task_type,
        input=input_text,
        model_answer=model_answer,
        reference=reference or "N/A"
    )
    
    raw = call_llm(SYSTEM_PROMPT, prompt)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": 0, "justification": "Invalid JSON from judge"}