import json
import os

from google import genai


LEGAL_CATEGORIES = [
    "עסקה ברוכלות",
    "עסקה מתמשכת",
    "עסקת מכר מרחוק",
    "ביטול עסקה בחנות",
    "לא ידוע",
]


def classify_legal_query(question: str) -> str:
    """
    Classify a consumer-law question into a legal category.

    Returns:
        One of LEGAL_CATEGORIES.
        Falls back to "לא ידוע" on failure.
    """

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    prompt = f"""
אתה מסווג שאלות בנושא זכויות צרכנים בישראל.

המטרה שלך היא לזהות את סוג העסקה המתואר בשאלה,
ולא לקבוע אם הצרכן זכאי לביטול.

בחר קטגוריה אחת בלבד מתוך הרשימה:
{json.dumps(LEGAL_CATEGORIES, ensure_ascii=False)}

כללי סיווג מחייבים:

1. "עסקה ברוכלות":
   בחר כאשר השאלה מתארת מכירה שהתבצעה בעקבות פנייה
   של מוכר לצרכן מחוץ לבית העסק, למשל בבית הצרכן.

2. "עסקה מתמשכת":
   בחר כאשר השאלה מתארת מנוי, שירות מתחדש,
   חיובים תקופתיים או בקשה להתנתק משירות מתמשך.

3. "עסקת מכר מרחוק":
   בחר כאשר השאלה מתארת רכישה באתר אינטרנט,
   בטלפון או באמצעי תקשורת מרחוק.

4. "ביטול עסקה בחנות":
   בחר רק כאשר השאלה מציינת במפורש שהרכישה
   התבצעה בחנות פיזית או בבית העסק.

5. "לא ידוע":
   בחר כאשר אין בשאלה מספיק מידע כדי לקבוע
   את סוג העסקה.

כללים חשובים:
- אל תניח שרכישה התבצעה בחנות אם הדבר לא נאמר.
- אל תנחש פרטים שהמשתמש לא סיפק.
- אל תבחר קטגוריה רק משום שהיא נפוצה.
- אם כמה קטגוריות אפשריות ואין דרך להכריע, בחר "לא ידוע".
- אל תמציא קטגוריות חדשות.

דוגמאות:

שאלה: "קניתי מוצר ואני רוצה להחזיר אותו"
תשובה: {{"legal_term": "לא ידוע"}}

שאלה: "רכשתי חולצה ואני רוצה לבטל את העסקה"
תשובה: {{"legal_term": "לא ידוע"}}

שאלה: "קניתי נעליים בחנות בקניון"
תשובה: {{"legal_term": "ביטול עסקה בחנות"}}

שאלה: "סוכן מכירות הגיע אליי הביתה ומכר לי מוצר"
תשובה: {{"legal_term": "עסקה ברוכלות"}}

החזר JSON בלבד בפורמט:
{{"legal_term": "שם הקטגוריה"}}

שאלת המשתמש:
{question}
"""

    try:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            contents=prompt,
            config={
                "temperature": 0,
                "response_mime_type": "application/json",
            },
        )

        data = json.loads(response.text)
        category = data.get("legal_term", "לא ידוע")

        if category not in LEGAL_CATEGORIES:
            return "לא ידוע"

        return category

    except Exception as exc:
        print(f"Classification failed: {exc}")
        return "לא ידוע"