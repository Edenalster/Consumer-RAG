import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

HISTORY_DIR = Path(__file__).resolve().parent / "data" / "chat_history"


def save_chat(chat_id, messages):
    """Save a conversation to a local JSON file."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    if not chat_id:
        chat_id = str(uuid4())

    # Use the first user question as the conversation title.
    first_question = next(
        (m["content"] for m in messages if m["role"] == "user"),
        "שיחה חדשה",
    )

    chat_path = HISTORY_DIR / f"{chat_id}.json"
    title = first_question[:60]

    if chat_path.exists():
        existing_chat = json.loads(
            chat_path.read_text(encoding="utf-8")
        )
        title = existing_chat.get("title") or title

    chat = {
        "id": chat_id,
        "title": title,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "messages": messages,
    }

    path = HISTORY_DIR / f"{chat_id}.json"
    temp_path = HISTORY_DIR / f"{chat_id}.tmp"
    temp_path.write_text(
        json.dumps(chat, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)

    return chat_id


def load_chat(chat_id):
    """Load a saved conversation."""
    path = HISTORY_DIR / f"{chat_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def list_chats():
    """List saved conversations, newest first."""
    if not HISTORY_DIR.exists():
        return []

    chats = []

    for path in HISTORY_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            chats.append({
                "id": data["id"],
                "title": data["title"],
                "updated_at": data["updated_at"],
            })
        except (OSError, ValueError, KeyError):
            continue

    return sorted(
        chats,
        key=lambda chat: chat["updated_at"],
        reverse=True,
    )


def delete_chat(chat_id):
    """Delete a saved conversation."""
    from uuid import UUID

    safe_id = str(UUID(chat_id))
    path = HISTORY_DIR / f"{safe_id}.json"

    if not path.exists():
        return False

    path.unlink()
    return True


def rename_chat(chat_id, new_title):
    """Rename a saved conversation without changing its messages."""
    from uuid import UUID

    safe_id = str(UUID(chat_id))
    title = new_title.strip()

    if not title:
        raise ValueError("Chat title cannot be empty")

    if len(title) > 100:
        raise ValueError("Chat title cannot exceed 100 characters")

    path = HISTORY_DIR / f"{safe_id}.json"

    if not path.exists():
        return False

    chat = json.loads(path.read_text(encoding="utf-8"))
    chat["title"] = title

    temp_path = HISTORY_DIR / f"{safe_id}.tmp"
    temp_path.write_text(
        json.dumps(chat, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)

    return True
