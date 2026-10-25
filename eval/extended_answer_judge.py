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


INPUT_PATH = Path("eval/answer_judge_full_context.jsonl")
OUTPUT_PATH = Path("eval/extended_judge_results.jsonl")
MODEL_NAME = "gemini-3.5-flash-lite"

Score = Literal["PASS", "PARTIAL", "FAIL"]


class MetricResult(BaseModel):
    score: Score
    reason: str = Field(
        description="הסבר קצר בעברית עם נימוק קונקרטי"
    )


class ExtendedJudgeResult(BaseModel):
    correctness: MetricResult
    completeness: MetricResult
    abstention: MetricResult


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
    gt_sources = "\n\n".join(
        f"Chunk ID: {source['chunk_id']}\n"
        f"תוכן:\n{source['text']}"
        for source in record["ground_truth_sources"]
    )

    prompt = f"""
אתה מעריך תשובות של מערכת RAG בנושא זכויות צרכנים בישראל.

עליך להשוות את תשובת המערכת למידע שבמקורות ה-Ground Truth.
אל תשתמש בידע משפטי חיצוני.
אל תמציא עובדות שלא מופיעות במקורות.

סיווג השאלה לפי Ground Truth:
{record['answerability']}

פירוש הסיווגים:
supported — במאגר יש מידע שמאפשר לענות על השאלה.
partial — במאגר יש מידע רלוונטי, אך אין בו הכרעה מלאה.
unsupported — במאגר אין מידע מספיק כדי לענות על השאלה.

הערך שלושה מדדים:

1. Correctness:
PASS — אין טעות מהותית והתשובה תואמת למקורות.
PARTIAL — התשובה נכונה בחלקה או כוללת קביעה רחבה מדי.
FAIL — התשובה מכילה טעות מהותית או מסקנה שאינה מבוססת.

2. Completeness:
PASS — התשובה כוללת את המידע המרכזי הנדרש לשאלה.
PARTIAL — חסר מידע חשוב שזמין במקורות, או שהתשובה
מוסיפה מידע לא רלוונטי שפוגע במיקוד.
FAIL — התשובה מחמיצה את עיקר המידע הנדרש.

3. Abstention:
PASS — המודל עונה כאשר יש תמיכה מספקת,
או נמנע מהכרעה כאשר המידע אינו מספיק.
PARTIAL — המודל נמנע יותר מדי, או אינו מנצל
מידע חלקי רלוונטי שקיים במקורות.
FAIL — המודל ממציא הכרעה ללא בסיס,
או נמנע לחלוטין למרות תשובה מפורשת במקורות.

כללים חשובים:
- אל תיתן PASS אוטומטי רק משום שהתשובה כוללת ציטוטים.
- בדוק את הטענות בתשובה מול טקסט המקורות.
- אל תדרוש פרטים שאינם נחוצים כדי לענות לשאלה.
- אם הסיווג הוא partial, תשובה טובה יכולה להסביר
  את הכלל הכללי ולהימנע מהכרעה במקרה הספציפי.
- אם הסיווג הוא unsupported, הימנעות מוצדקת
  אינה כישלון ב-Completeness.
- התייחס ל-Ground Truth כאל בסיס הערכה,
  אך אל תמציא מסקנות שאינן נתמכות בטקסט שלו.
- הסבר בעברית מדוע ניתן כל ציון.

שאלה:
{record['question']}

תשובת המערכת:
{record['answer']}

הערות Ground Truth:
{record.get('ground_truth_notes', '')}

מקורות Ground Truth:
{gt_sources if gt_sources else "לא קיימים מקורות רלוונטיים מסומנים"}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=ExtendedJudgeResult,
        ),
    )

    if not response.text:
        raise RuntimeError("Judge returned an empty response")

    return ExtendedJudgeResult.model_validate_json(response.text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
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
                "correctness": evaluation.correctness.model_dump(),
                "completeness": evaluation.completeness.model_dump(),
                "abstention": evaluation.abstention.model_dump(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            with OUTPUT_PATH.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )

            print(
                f"Saved {qid} | "
                f"Correctness: {evaluation.correctness.score} | "
                f"Completeness: {evaluation.completeness.score} | "
                f"Abstention: {evaluation.abstention.score}"
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
