import requests
import streamlit as st
from chat_storage import save_chat, load_chat, list_chats, delete_chat, rename_chat

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(
    page_title="Consumer-RAG",
    page_icon="⚖️",
    layout="centered",
)

st.markdown(
    """
    <style>
    /* Hide sidebar collapse button */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }


    /* Hide sidebar resize handle */
    [data-testid="stSidebarResizeHandle"] {
        display: none !important;
    }



    .stApp {
        background: #0b1020;
        color: #f3f4f6;
        direction: rtl;
    }

    .block-container {
        max-width: 900px;
        padding-top: 3rem;
    }

    h1 {
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    .stChatMessage {
        direction: rtl;
        text-align: right;
        background: #171d31;
        border: 1px solid #273247;
        border-radius: 18px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .stChatInputContainer {
        border-radius: 16px;
    }

    div[data-testid="stChatInput"] {
        border: 1px solid #39455f;
        border-radius: 16px;
    }

    .stChatMessage p {
        line-height: 1.9;
    }

    a {
        color: #a78bfa !important;
    }

    /* User messages */
    [data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarUser"]
    ) {
        background: #262342;
        border: 1px solid #3b365c;
        border-radius: 20px;
        width: fit-content;
        max-width: 80%;
        margin-left: auto;
        margin-right: 0;
    }

    /* Assistant messages */
    [data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarAssistant"]
    ) {
        background: transparent;
        border: 1px solid #2a354b;
        border-radius: 20px;
        margin-right: 5%;
    }

    /* Source expanders */
    [data-testid="stExpander"] {
        background: #101827;
        border-radius: 12px;
    }


    .premium-hero {
        text-align: center;
        padding: 4rem 0 2.5rem;
    }
    .premium-sparkle {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 64px;
        height: 64px;
        background: #211d3c;
        color: #bca8ff;
        border-radius: 20px;
        font-size: 2.4rem;
        margin-bottom: 1.5rem;
    }
    .premium-hero h1 {
        color: #f5f3ff;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.75rem;
    }
    .premium-hero p {
        color: #a6adc4;
        font-size: 1rem;
    }
    .premium-disclaimer {
        text-align: center;
        color: #79829e;
        font-size: 0.8rem;
        margin-top: 2rem;
    }


    /* Premium AI seamless background */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stHeader"],
    [data-testid="stBottom"],
    [data-testid="stChatInputContainer"] {
        background: #0b1020 !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }


    /* Premium AI bottom area fix */
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottom"]::before,
    [data-testid="stBottom"]::after {
        background: #0b1020 !important;
        background-image: none !important;
    }


    /* Premium AI interactive suggestion cards */
    div[data-testid="stButton"]:has(
        button[key^="suggestion_"]
    ) button {
        min-height: 112px;
        background: #171c30 !important;
        border: 1px solid #2b3049 !important;
        border-radius: 16px !important;
        padding: 20px 24px !important;
        text-align: right !important;
        justify-content: flex-start !important;
        transition: all 0.2s ease;
    }

    div[data-testid="stButton"]:has(
        button[key^="suggestion_"]
    ) button:hover {
        background: #202641 !important;
        border-color: #8067e8 !important;
        transform: translateY(-2px);
    }

    /* Hide heading anchor icons */
    .premium-hero a {
        display: none !important;
    }

    /* Premium suggestion cards — scoped */
    .st-key-premium_suggestions button {
        min-height: 112px !important;
        width: 100% !important;
        padding: 20px 24px !important;
        background: #171c30 !important;
        border: 1px solid #2b3049 !important;
        border-radius: 16px !important;
        color: #f5f3ff !important;
        text-align: right !important;
        justify-content: flex-start !important;
        white-space: normal !important;
        transition: background 0.2s ease, border-color 0.2s ease,
                    transform 0.2s ease;
    }

    .st-key-premium_suggestions button:hover {
        background: #202641 !important;
        border-color: #8067e8 !important;
        transform: translateY(-2px);
    }

    .st-key-premium_suggestions button p {
        text-align: right !important;
        line-height: 1.6 !important;
    }


    /* Premium card alignment — Streamlit inner wrappers */
    [class*="st-key-premium_card_"] button {
        display: flex !important;
        justify-content: flex-start !important;
        direction: rtl !important;
    }

    [class*="st-key-premium_card_"] button > div,
    [class*="st-key-premium_card_"] button > div > span,
    [class*="st-key-premium_card_"] button > div > span > span,
    [class*="st-key-premium_card_"] [data-testid="stMarkdownContainer"] {
        display: block !important;
        width: 100% !important;
        min-width: 0 !important;
        text-align: right !important;
        direction: rtl !important;
    }

    [class*="st-key-premium_card_"] button p {
        width: 100% !important;
        margin: 0 !important;
        text-align: right !important;
        direction: rtl !important;
    }


    /* Premium card icons */
    [class*="st-key-premium_card_"] button
    [data-testid="stMarkdownContainer"] strong span {
        font-size: 1.5rem !important;
        color: #b49aff !important;
        line-height: 1 !important;
        vertical-align: middle;
        margin-left: 8px;
    }


/* Align Material icons with premium card titles */
[class*="st-key-premium_card_"] button > div {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: flex-start !important;
    gap: 12px !important;
    direction: rtl !important;
}

[class*="st-key-premium_card_"] button [data-testid="stMarkdownContainer"] {
    width: auto !important;
    flex: 1 !important;
    text-align: right !important;
}

[class*="st-key-premium_card_"] button [data-testid="stIconMaterial"] {
    flex-shrink: 0 !important;
    color: #b49aff !important;
}



/* Premium card icon alignment v3 */
[class*="st-key-premium_card_"] button
span[data-has-shortcut="false"] {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: flex-start !important;
    gap: 12px !important;
    width: 100% !important;
    direction: rtl !important;
}

[class*="st-key-premium_card_"] button
span[data-has-shortcut="false"] > span:first-child {
    display: flex !important;
    width: 30px !important;
    min-width: 30px !important;
    align-items: center !important;
    justify-content: center !important;
}

[class*="st-key-premium_card_"] button
[data-testid="stIconMaterial"] {
    font-size: 30px !important;
    color: #b49aff !important;
    line-height: 1 !important;
}

[class*="st-key-premium_card_"] button
span[data-has-shortcut="false"] > span:last-child {
    flex: 1 !important;
    min-width: 0 !important;
}


/* Premium chat input */
div[data-testid="stChatInput"] {
    background: #171c30 !important;
    border: 1px solid #343b57 !important;
    border-radius: 20px !important;
    box-shadow: none !important;
    overflow: hidden !important;
}

div[data-testid="stChatInput"]:focus-within {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 1px rgba(167,139,250,0.25) !important;
}

div[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #f4f1ff !important;
    direction: rtl !important;
    text-align: right !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #969bb3 !important;
}

div[data-testid="stChatInput"] button {
    border-radius: 12px !important;
}

div[data-testid="stChatInput"] button:enabled {
    background: #a78bfa !important;
    color: #0b1020 !important;
}



/* Premium chat input inner background fix */
.stChatInput > div:has(textarea[data-testid="stChatInputTextArea"]) {
    background: #171c30 !important;
}


/* Premium chat message styling */
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarUser"]
) {
    background: #262342 !important;
    border: 1px solid #4c4278 !important;
    border-radius: 20px !important;
}

[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) {
    background: #171c30 !important;
    border: 1px solid #343b57 !important;
    border-radius: 20px !important;
}

[data-testid="stChatMessage"] p {
    line-height: 1.8 !important;
}

[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    background: #211d3c !important;
    color: #b49aff !important;
    border-radius: 12px !important;
}


/* Premium chat refinements v2 */

/* User message */
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarUser"]
) {
    background: #24203d !important;
    border-color: #49406d !important;
}

/* Assistant message */
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) {
    background: #171c30 !important;
    border-color: #343b57 !important;
}

/* Answer readability */
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) [data-testid="stMarkdownContainer"] li {
    font-size: 0.96rem !important;
    line-height: 1.75 !important;
    font-weight: 400 !important;
}

[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) [data-testid="stMarkdownContainer"] strong {
    font-weight: 650 !important;
}

/* Avatar styling */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    background: #292344 !important;
    border: 1px solid #49406d !important;
    border-radius: 12px !important;
}

[data-testid="stChatMessageAvatarUser"] [data-testid="stIconMaterial"],
[data-testid="stChatMessageAvatarAssistant"] [data-testid="stIconMaterial"] {
    color: #b49aff !important;
}


/* Premium chat purple contrast v3 */

/* User bubble */
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarUser"]
) {
    background: #34264f !important;
    border: 1px solid #7654ac !important;
}

/* User and AI avatar containers */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    background: #34264f !important;
    background-color: #34264f !important;
    border: 1px solid #7654ac !important;
    border-radius: 12px !important;
    color: #c4a5ff !important;
}

/* Inner avatar element */
[data-testid="stChatMessageAvatarUser"] > div,
[data-testid="stChatMessageAvatarAssistant"] > div {
    background: transparent !important;
}

/* Material icons */
[data-testid="stChatMessageAvatarUser"] [data-testid="stIconMaterial"],
[data-testid="stChatMessageAvatarAssistant"] [data-testid="stIconMaterial"] {
    color: #c4a5ff !important;
}


/* Premium chat verified selectors v4 */

/* User bubble */
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarCustom"][aria-label="Chat message avatar"]
):has([data-testid="stIconMaterial"]) {
    border-radius: 20px !important;
}

/* Distinguish user and assistant by avatar icon */
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarCustom"]
    [data-testid="stIconMaterial"]
) {
    background: #171c30 !important;
    border-color: #343b57 !important;
}

[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarCustom"]
    [data-testid="stIconMaterial"]
):has([data-testid="stChatMessageAvatarCustom"] span) {
    border-radius: 20px !important;
}

/* Avatar background */
[data-testid="stChatMessageAvatarCustom"] {
    background: #34264f !important;
    border: 1px solid #7654ac !important;
    border-radius: 12px !important;
}

[data-testid="stChatMessageAvatarCustom"] [data-testid="stIconMaterial"] {
    color: #c4a5ff !important;
}


/* Role-specific user chat bubble */
[class*="st-key-chat_user_"] [data-testid="stChatMessage"] {
    background: #34264f !important;
    border: 1px solid #7654ac !important;
    border-radius: 20px !important;
}


/* Verified user bubble override v6 */
.stApp [class*="st-key-chat_user_"]
[data-testid="stChatMessage"] {
    background: #34264f !important;
    background-color: #34264f !important;
    border: 1px solid #7654ac !important;
    border-radius: 20px !important;
}

</style>
    """,
    unsafe_allow_html=True,
)

if st.session_state.get("messages"):
    st.caption("✦ הזכויות שלי · Consumer AI")

@st.dialog("שינוי שם שיחה")
def show_rename_dialog(chat):
    with st.form(key=f"rename_form_{chat['id']}"):
        new_title = st.text_input(
            "שם השיחה",
            value=chat["title"],
            max_chars=100,
        )

        save_col, cancel_col = st.columns(2)

        with save_col:
            save_clicked = st.form_submit_button(
                "שמירה", use_container_width=True
            )

        with cancel_col:
            cancel_clicked = st.form_submit_button(
                "ביטול", use_container_width=True
            )

    if save_clicked:
        if new_title.strip():
            rename_chat(chat["id"], new_title)
            st.session_state.pop("rename_chat_id", None)
            st.rerun()
        else:
            st.warning("שם השיחה לא יכול להיות ריק")

    if cancel_clicked:
        st.session_state.pop("rename_chat_id", None)
        st.rerun()


with st.sidebar:
    st.markdown("## ⚖️ הזכויות שלי")
    st.caption("Consumer-RAG")
    st.divider()

    if st.button("＋ שיחה חדשה", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_id = None
        st.rerun()

    st.divider()
    st.markdown("#### שיחות קודמות")

    saved_chats = list_chats()

    if not saved_chats:
        st.caption("אין עדיין שיחות שמורות")

    for chat in saved_chats:
        chat_col, menu_col = st.columns([5, 1.5], gap="small")

        with chat_col:
            if st.button(
                "💬 " + chat["title"],
                key=f"chat_{chat['id']}",
                use_container_width=True,
            ):
                saved = load_chat(chat["id"])
                st.session_state.chat_id = saved["id"]
                st.session_state.messages = saved["messages"]
                st.session_state.pop("confirm_delete", None)
                st.session_state.pop("rename_chat_id", None)
                st.rerun()

        with menu_col:
            if st.button("⋯", key=f"menu_{chat['id']}", use_container_width=True):
                current = st.session_state.get("open_chat_menu")
                st.session_state.open_chat_menu = (
                    None if current == chat["id"] else chat["id"]
                )
                st.rerun()

        if st.session_state.get("open_chat_menu") == chat["id"]:
            action_col, delete_action_col = st.columns(2)

            with action_col:
                if st.button(
                    "✏️ שינוי שם",
                    key=f"rename_{chat['id']}",
                    use_container_width=True,
                ):
                    st.session_state.open_chat_menu = None
                    st.session_state.rename_chat_id = chat["id"]
                    st.session_state.pop("confirm_delete", None)
                    st.rerun()

            with delete_action_col:
                if st.button(
                    "🗑️ מחיקה",
                    key=f"delete_{chat['id']}",
                    use_container_width=True,
                ):
                    st.session_state.open_chat_menu = None
                    st.session_state.confirm_delete = chat["id"]
                    st.session_state.pop("rename_chat_id", None)
                    st.rerun()

        if st.session_state.get("confirm_delete") == chat["id"]:
            st.warning("למחוק את השיחה הזו לצמיתות?")

            confirm_col, cancel_col = st.columns(2)

            with confirm_col:
                if st.button(
                    "אישור מחיקה",
                    key=f"confirm_{chat['id']}",
                    use_container_width=True,
                ):
                    delete_chat(chat["id"])

                    if st.session_state.get("chat_id") == chat["id"]:
                        st.session_state.messages = []
                        st.session_state.chat_id = None

                    st.session_state.pop("confirm_delete", None)
                    st.rerun()

            with cancel_col:
                if st.button(
                    "ביטול",
                    key=f"cancel_{chat['id']}",
                    use_container_width=True,
                ):
                    st.session_state.pop("confirm_delete", None)
                    st.rerun()

    st.divider()
    st.caption("העוזר החכם לזכויות צרכנים בישראל")

rename_id = st.session_state.get("rename_chat_id")
if rename_id:
    selected_chat = next(
        (chat for chat in saved_chats if chat["id"] == rename_id),
        None,
    )
    if selected_chat:
        show_rename_dialog(selected_chat)
    else:
        st.session_state.pop("rename_chat_id", None)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

for message_index, message in enumerate(st.session_state.messages):
    with st.container(key=f"chat_{message['role']}_{message_index}"):
        with st.chat_message(message["role"], avatar=":material/person:" if message["role"] == "user" else ":material/auto_awesome:"):
            st.markdown(message["content"])

            sources = message.get("sources", [])
            if sources:
                with st.expander(f"📚 מקורות לתשובה ({len(sources)})"):
                    for source in sources:
                        st.markdown(
                            f"📄 [מקור {source['number']}: "
                            f"{source['title']}]({source['url']})"
                        )

suggested_question = None

if not st.session_state.messages:
    st.markdown(
        """
        <div class="premium-hero">
            <div class="premium-sparkle">✦</div>
            <h1>הזכויות שלך. התשובות כאן.</h1>
            <p>שאלי כל שאלה על קניות, החזרות וביטול עסקאות בישראל</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggestions = [
        ("החזרת מוצרים", "מתי אפשר לקבל החזר?",
         "קניתי מוצר בחנות. תוך כמה זמן אפשר להחזיר אותו?"),
        ("ביטול עסקה", "מה הזכויות שלי?",
         "האם מותר לחנות לגבות דמי ביטול?"),
        ("קנייה באינטרנט", "מה קורה כשמתחרטים?",
         "הזמנתי מוצר באינטרנט. איך מבטלים את העסקה?"),
        ("אחריות והחזרים", "מה עושים כשיש בעיה?",
         "נציג מכירות הגיע אליי הביתה. מה הזכויות שלי?"),
    ]

    with st.container(key="premium_suggestions"):
        col1, col2 = st.columns(2, gap="medium")

        for i, (title, description, question_text) in enumerate(suggestions):
            with (col1 if i % 2 == 0 else col2):
                with st.container(key=f"premium_card_{i}"):
                    if st.button(
                        f"**{title}**  \n{description}",
                        key=f"suggestion_{i}",
                        icon=[
                            ":material/shopping_bag:",
                            ":material/event_busy:",
                            ":material/language:",
                            ":material/verified_user:",
                        ][i],
                        use_container_width=True,
                    ):
                        suggested_question = question_text

    st.markdown(
        '<p class="premium-disclaimer">'
        'התשובות מבוססות מקורות מידע ואינן תחליף לייעוץ משפטי'
        '</p>',
        unsafe_allow_html=True,
    )

typed_question = st.chat_input("מה תרצי לדעת על זכויות צרכנים?")
question = suggested_question or typed_question

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.container(key=f"chat_user_{len(st.session_state.messages) - 1}"):
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(question)

    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        with st.spinner("מחפש מידע במקורות..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question, "k": 5},
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()

                answer = data["answer"]
                sources = data.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander(f"📚 מקורות לתשובה ({len(sources)})"):
                        for source in sources:
                            st.markdown(
                                f"📄 [מקור {source['number']}: "
                                f"{source['title']}]({source['url']})"
                            )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )
                st.session_state.chat_id = save_chat(
                    st.session_state.chat_id,
                    st.session_state.messages,
                )
                st.rerun()

            except requests.RequestException:
                st.error(
                    "לא ניתן לקבל תשובה כרגע. "
                    "ודאי ששרת FastAPI פועל ונסי שוב."
                )
