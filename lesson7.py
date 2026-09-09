import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)


api_key = st.secrets["GEMINI_API_KEY"]

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key
)


if "history" not in st.session_state:
    st.session_state.history = [
        SystemMessage(
            """
            Ти — чат-бот для допомоги у вивченні англійської мови.

            Твоя задача — допомагати користувачу з перекладом
            англійських слів, фраз та речень.

            Якщо користувач просить перекласти слово або коротку фразу:
            1. Напиши переклад українською мовою.
            2. Додай приклад використання цього слова або фрази
               в англійському реченні.
            3. Додай переклад прикладу українською мовою.

            Якщо користувач просить перекласти речення:
            1. Напиши переклад українською мовою.
            2. Поясни граматику речення.
            3. Вкажи граматичну структуру, яка використовується.
               Наприклад:
               - there is / there are
               - Present Simple
               - Past Simple
               - пасивна форма дієслова
               - умовні речення
               - модальні дієслова
               - інші граматичні конструкції.

            Пояснюй граматику просто і зрозуміло.

            Якщо користувач ставить звичайне питання
            про англійську мову, також допомагай йому.

            Відповідай українською мовою.
            """
        )
    ]


st.title("Помічник з вивчення англійської")

user_text = st.chat_input(
    "Напишіть слово, фразу або речення"
)

if user_text is not None:
    human_message = HumanMessage(
        content=user_text
    )

    messages = st.session_state.history

    messages.append(human_message)

    response = llm.invoke(messages)

    messages.append(
        AIMessage(content=response.content)
    )


for message in st.session_state.history:
    if isinstance(message, SystemMessage):
        continue

    if isinstance(message, HumanMessage):
        role = "user"
    else:
        role = "assistant"

    with st.chat_message(role):
        st.markdown(message.content)

