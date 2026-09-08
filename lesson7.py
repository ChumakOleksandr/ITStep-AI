import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)
# Завдання 1
# Напишіть додаток, який симулює спілкування з певною
# відомою людиною.
# З ким саме спілкуватись вводить користувач через
# st.text_input()

# api_key = st.secrets["GEMINI_API_KEY"]
#
# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.5-flash-lite",
#     api_key=api_key
# )
#
# st.title("Чат з відомою людиною")
#
# person = st.text_input(
#     "З ким ви хочете поспілкуватися?",
#     placeholder="Наприклад: Ілон Маск"
# )
#
# if person and "history" not in st.session_state:
#     st.session_state.history = [
#         SystemMessage(
#             f"""
#             Ти симулюєш спілкування з відомою людиною — {person}.
#
#             Веди себе так, ніби ти ця людина.
#             Відповідай у її стилі, враховуй її професію,
#             відомі факти про неї та особливості її характеру.
#
#             Не пояснюй, що ти штучний інтелект.
#             Підтримуй звичайне спілкування з користувачем.
#             """
#         )
#     ]
#
# if person and st.session_state.get("current_person") != person:
#     st.session_state.current_person = person
#     st.session_state.history = [
#         SystemMessage(
#             f"""
#             Ти симулюєш спілкування з відомою людиною — {person}.
#
#             Веди себе так, ніби ти ця людина.
#             Відповідай у її стилі та враховуй відомі факти
#             про її життя, професію та діяльність.
#
#             Підтримуй природне спілкування з користувачем.
#             """
#         )
#     ]
#
# if not person:
#     st.info("Спочатку введіть ім'я відомої людини")
#
# user_text = st.chat_input("Введіть повідомлення")
#
# if user_text is not None and person:
#
#     human_message = HumanMessage(content=user_text)
#
#     messages = st.session_state.history
#
#     messages.append(human_message)
#
#     response = llm.invoke(messages)
#
#     messages.append(response)
#
# if "history" in st.session_state:
#
#     for message in st.session_state.history:
#         if isinstance(message, SystemMessage):
#             continue
#
#         if isinstance(message, HumanMessage):
#             role = "user"
#         else:
#             role = "assistant"
#
#         with st.chat_message(role):
#             st.markdown(message.content)

# Завдання 2
# Напишіть додаток, який симулює проходження
# співбесіди на певну посаду.
# Користувач може ввести назву посади через st.text_input()
# Користувач може ввести опис вакансії через
# st.file_uploader()
# Далі починається чат з спілкуванням

api_key = st.secrets["GEMINI_API_KEY"]

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key
)

st.title("Співбесіда")

position = st.text_input(
    "Введіть назву посади",
    placeholder="Наприклад: Python Developer"
)

uploaded_file = st.file_uploader(
    "Завантажте опис вакансії",
    type=["txt", "pdf", "docx"]
)

vacancy_text = ""

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        vacancy_text = uploaded_file.read().decode("utf-8")

    else:
        st.info("Для цього прикладу використовуйте .txt файл")


if position and vacancy_text:
    if "history" not in st.session_state:
        st.session_state.history = [
            SystemMessage(
                f"""
                Ти — HR-менеджер, який проводить співбесіду
                на посаду {position}.

                Опис вакансії:
                {vacancy_text}

                Проведи співбесіду з кандидатом.

                Став питання по одному.
                Питання повинні відповідати посаді та опису вакансії.
                Запитуй про досвід, знання та навички кандидата.

                Після кожної відповіді кандидата
                став наступне питання.

                Не став декілька питань одночасно.
                """
            )
        ]

        st.session_state.history.append(
            AIMessage(
                content=f"Вітаю! Почнемо співбесіду на посаду {position}. Розкажіть трохи про себе та свій досвід."
            )
        )


if not position:
    st.info("Введіть назву посади")

elif not vacancy_text:
    st.info("Завантажте опис вакансії у форматі .txt")


user_text = st.chat_input("Ваша відповідь")


if user_text is not None and position and vacancy_text:
    human_message = HumanMessage(content=user_text)

    messages = st.session_state.history

    messages.append(human_message)

    response = llm.invoke(messages)

    messages.append(response)


if "history" in st.session_state:
    for message in st.session_state.history:
        if isinstance(message, SystemMessage):
            continue

        if isinstance(message, HumanMessage):
            role = "user"
        else:
            role = "assistant"

        with st.chat_message(role):
            st.markdown(message.content)

