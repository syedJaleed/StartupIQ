# analyst_risk_ui_with_metrics.py
# Generate Risk Assessment + AI Recommendations + Metrics with Gemini (JSON-only).
# Requires: google-genai, python-dotenv
#   pip install google-genai python-dotenv

from google import genai
from dotenv import load_dotenv
import os, json, argparse, uuid, datetime, sys
from pathlib import Path
from typing import Dict, Any

# ---------------------- Config ----------------------

DEFAULT_MODEL = os.getenv("GENAI_MODEL", "gemini-2.0-flash-lite-001")
DEFAULT_PROJECT = os.getenv("GENAI_PROJECT", "test-project-471516")
DEFAULT_LOCATION = os.getenv("GENAI_LOCATION", "us-central1")
OUTDIR = Path(os.getenv("ANALYST_OUTDIR", "runs"))
OUTDIR.mkdir(parents=True, exist_ok=True)

GENERATION = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_output_tokens": 1600,
    "candidate_count": 1
}

SYSTEM_PROMPT = r"""
You are a pragmatic startup analyst. Return ONLY valid JSON with this exact schema (no extra keys, no markdown):

{
  "risk_assessment": {
    "key_strengths": ["", ""],
    "risk_factors": ["", ""]
  },
  "ai_recommendations": [
    {
      "area": "Product|Business|Go-To-Market|Ops|Finance",
      "title": "Short actionable recommendation (<= 20 words)",
      "priority": "HIGH|MEDIUM|LOW",
      "impact": "One line describing expected impact (<= 20 words)"
    }
  ],
  "metrics": [
    {
      "name": "Problem Severity",
      "score": 0,
      "reason": "1 sentence on why"
    },
    {
      "name": "Solution Uniqueness",
      "score": 0,
      "reason": "1 sentence on why"
    },
    {
      "name": "Idea Viability",
      "score": 0,
      "reason": "1 sentence on why"
    },
    {
      "name": "Technical Feasibility",
      "score": 0,
      "reason": "1 sentence on why"
    }
  ]
}

Rules:
- Keep bullets concise and concrete; avoid fluff.
- Prefer 4–6 strengths and 4–6 risks when possible.
- For recommendations: 2–4 items total; each must have area, priority, and impact.
- For metrics: use integer scores 1–10 (10 = excellent).
- Output must be VALID JSON only.
"""

# ---------------------- Input builder ----------------------

def make_user_prompt(idea: Dict[str, Any]) -> str:
    return f"""
Startup Idea (free text):
{idea.get('idea_text', '').strip()}

Context (assume accurate):
- Target market: {idea.get('target_market', '')}
- Existing alternatives: {idea.get('alternatives', '')}
- Pricing: {idea.get('pricing', '')}
- Team: {idea.get('team', '')}

Task:
Analyze the startup and produce the JSON per schema (risk_assessment + ai_recommendations + metrics) only.
""".strip()

# ---------------------- Utilities ----------------------

def parse_args():
    p = argparse.ArgumentParser(description="Gemini: Risk Assessment + AI Recommendations + Metrics (UI JSON)")
    p.add_argument("--input", type=str, default="", help="Path to idea.json (optional)")
    p.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    p.add_argument("--location", type=str, default=DEFAULT_LOCATION)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    return p.parse_args()

def load_input_or_default(path: str) -> Dict[str, Any]:
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Sensible default if no file given
    return {
        "idea_text": "Scheduling assistant for grocery retail that auto-builds weekly shift rosters from demand forecasts and posts to WhatsApp for approvals.",
        "target_market": "Frontline retail managers in Indian grocery chains (50–500 employees per store)",
        "alternatives": "Excel rosters, WhatsApp messages, manual copy-paste, basic POS exports",
        "pricing": "SaaS $199/store/month",
        "team": "Ex-Flipkart ops lead + ML engineer with workforce planning experience"
    }

def extract_resp_text(resp) -> str:
    if hasattr(resp, "text") and isinstance(resp.text, str):
        return resp.text
    try:
        return resp.candidates[0].content.parts[0].text
    except Exception:
        return str(resp)

def validate_ui_schema(data: Dict[str, Any]) -> None:
    # Top-level
    for key in ["risk_assessment", "ai_recommendations", "metrics"]:
        if key not in data:
            raise ValueError(f"Missing top-level key: {key}")
    # Risk assessment
    ra = data["risk_assessment"]
    for arr in ["key_strengths", "risk_factors"]:
        if arr not in ra or not isinstance(ra[arr], list):
            raise ValueError(f"risk_assessment must include array: {arr}")
    # Recs
    recs = data["ai_recommendations"]
    if not isinstance(recs, list) or len(recs) == 0:
        raise ValueError("ai_recommendations must be a non-empty array")
    for rec in recs:
        for f in ["area", "title", "priority", "impact"]:
            if f not in rec:
                raise ValueError(f"Recommendation missing field: {f}")
    # Metrics
    mets = data["metrics"]
    if not isinstance(mets, list) or len(mets) < 4:
        raise ValueError("metrics must include at least 4 items")
    for m in mets:
        for f in ["name", "score", "reason"]:
            if f not in m:
                raise ValueError(f"Metric missing field: {f}")

# ---------------------- Main ----------------------

def main():
    load_dotenv()
    args = parse_args()

    client = genai.Client(
        vertexai=True,
        project=args.project,
        location=args.location
    )

    idea = load_input_or_default(args.input)
    user_prompt = make_user_prompt(idea)

    resp = client.models.generate_content(
        model=args.model,
        contents=user_prompt,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "response_mime_type": "application/json",
            "temperature": GENERATION["temperature"],
            "top_p": GENERATION["top_p"],
            "max_output_tokens": GENERATION["max_output_tokens"],
            "candidate_count": GENERATION["candidate_count"]
        }
    )

    raw = extract_resp_text(resp)

    try:
        result = json.loads(raw)
    except Exception:
        print("⚠️ Model did not return valid JSON. Raw response below:\n", file=sys.stderr)
        print(raw, file=sys.stderr)
        raise

    validate_ui_schema(result)

    # File outputs
    run_id = uuid.uuid4().hex
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    json_path = OUTDIR / f"{run_id}_ui_result.json"
    props_path = OUTDIR / f"{run_id}_risk_panel_props.json"

    artifact = {
        "ts": ts,
        "run_id": run_id,
        "model": args.model,
        "input": idea,
        "output": result
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n🗂  Saved UI JSON: {json_path}")

if __name__ == "__main__":
    main()