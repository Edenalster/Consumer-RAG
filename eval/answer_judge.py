import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


INPUT_PATH = Path("eval/answer_evaluation_context.jsonl")
OUTPUT_PATH = Path("eval/answer_judge_results.jsonl")
MODEL_NAME = "gemini-3.5-flash-lite"

Score = Literal["PASS", "PARTIAL", "FAIL"]


class MetricResult(BaseModel):
    score: Score
    reason: str = Field(
        description="הסבר קצר בעברית המבוסס על המקורות"
    )


class JudgeResult(BaseModel):
    faithfulness: MetricResult
    citation_accuracy: MetricResult


def read_jsonl(path):
    if not path.exists():
        return []

    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def evaluate(client, record):
    sources_text = "\n\n".join(
        f"[מקור {source['number']}]\n"
        f"Chunk ID: {source['chunk_id']}\n"
        f"תוכן:\n{source['text']}"
        for source in record["sources"]
    )

    prompt = f"""
אתה מעריך איכות של מערכת RAG בנושא זכויות צרכנים בישראל.

הערך את התשובה אך ורק על סמך המקורות שסופקו.
אל תשתמש בידע משפטי חיצוני.

מדדים:

1. Faithfulness:
PASS — כל הטענות העובדתיות בתשובה נתמכות במקורות.
PARTIAL — קיימות טענות עם תמיכה לא מספקת או ניסוח רחב מדי.
FAIL — קיימת טענה מהותית שסותרת את המקורות או אינה נתמכת.

2. Citation Accuracy:
PASS — ההפניות מדויקות ותומכות בטענות הרלוונטיות.
PARTIAL — קיימות הפניות מיותרות, כלליות או חלקיות.
FAIL — הפניות מהותיות שגויות, חסרות או מפנות למידע שאינו תומך בטענה.

כללים:
- אל תעניש תשובה רק משום שהיא קצרה.
- אל תעניש הימנעות מתשובה אם אין מספיק מידע.
- בדוק כל טענה מול המקור שמצוטט לצידה.
- הימנעות מהכרעה אינה הופכת טענות אחרות למבוססות.
- כאשר התשובה טוענת שהמקורות אינם מספיקים,
  אל תניח שהטענה נכונה רק משום שהמודל אמר זאת.
- אם מקור מצוטט אינו תומך בטענה, ציין זאת בהסבר.
- החזר הסברים קצרים בעברית.

שאלה:
{record['question']}

תשובת המערכת:
{record['answer']}

מקורות מצוטטים:
{sources_text if sources_text else "לא צוטטו מקורות"}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=JudgeResult,
        ),
    )

    if not response.text:
        raise RuntimeError("Judge returned an empty response")

    return JudgeResult.model_validate_json(response.text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be zero or greater")

    load_dotenv(dotenv_path=".env")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")

    records = read_jsonl(INPUT_PATH)
    previous = read_jsonl(OUTPUT_PATH)

    completed = {
        record["id"]
        for record in previous
        if record.get("status") == "success"
        and record.get("judge_model") == MODEL_NAME
    }

    pending = [
        record
        for record in records
        if record["id"] not in completed
    ][:args.limit]

    print("Total answers:", len(records))
    print("Already evaluated:", len(completed))
    print("Evaluating now:", len(pending))

    if not pending:
        print("Nothing to evaluate.")
        return

    client = genai.Client(api_key=api_key)

    for record in pending:
        qid = record["id"]
        print(f"\nEvaluating {qid}...")

        try:
            evaluation = evaluate(client, record)

            result = {
                "id": qid,
                "judge_model": MODEL_NAME,
                "status": "success",
                "faithfulness": evaluation.faithfulness.model_dump(),
                "citation_accuracy": evaluation.citation_accuracy.model_dump(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            with OUTPUT_PATH.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )

            print(
                f"Saved {qid} | "
                f"Faithfulness: {evaluation.faithfulness.score} | "
                f"Citations: {evaluation.citation_accuracy.score}"
            )

        except Exception as error:
            print(
                f"FAILED {qid}: "
                f"{type(error).__name__}: {error}"
            )
            raise

    print("\nDONE")
    print("Results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
