import dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.prompts import PromptTemplate


dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key
)

# Завдання 1
# Напишіть чат модель яка підсумовує всю розмову в
# декілька речень. Вкажіть щоб модель зберігала якомога
# більше деталей.
# Використайте цю модель для простого чат бота який
# замість trim_massages використовує модель з підсумуванням.
# Підсумовуйте повідомлення, коли їх більше 4.
# Старі повідомлення треба видалити
# НЕ ВИДАЛЯТИ SystemMessage та не використовувати
# його для підсумування

summary_prompt = PromptTemplate.from_template("""
    Ти -- модель для підсумовування історії розмови.
    Твоя задача -- підсумувати всю розмову у декілька речень.

    ###ІНСТРУКЦІЇ###

    1. Збережи якомога більше важливих деталей розмови.
    2. Не втрачати важливу інформацію, яку повідомив користувач.
    3. Збережи факти, імена, числа, дати, плани,
       вподобання та інші важливі деталі.
    4. Збережи основні питання користувача
       та відповіді чатбота.
    5. Підсумок має бути коротким, але максимально
       інформативним.
    6. Не вигадуй інформацію.
    7. Пиши підсумок у декількох реченнях.

    ###ІСТОРІЯ РОЗМОВИ###

    {conversation}

    ###ПІДСУМОК###
""")

summary_chain = summary_prompt | llm

system_message = SystemMessage("""
    Ти -- ввічливий чатбот.
    Твоя задача підтримувати спілкування з користувачем.

    ###ІНСТРУКЦІЇ###

    1. Відповідай коротко та зрозуміло.
    2. Враховуй попередню історію розмови.
    3. Якщо в історії є підсумок попередньої розмови,
       використовуй його як контекст.
""")

messages = [
    system_message
]

chain = llm

while True:
    user_text = input("Ви: ")

    if user_text == "":
        break

    human_message = HumanMessage(
        content=user_text
    )

    messages.append(human_message)


    message_count = len(messages) - 1

    if message_count > 4:
        print()
        print("Підсумовую історію розмови...")

        conversation_messages = messages[1:]

        conversation = ""

        for message in conversation_messages:
            if isinstance(message, HumanMessage):
                conversation += "Human: "
                conversation += message.content
                conversation += "\n"

            elif isinstance(message, AIMessage):
                conversation += "AI: "
                conversation += message.content[0]["text"]
                conversation += "\n"

        data = {
            "conversation": conversation
        }

        summary_response = summary_chain.invoke(data)

        summary_text = summary_response.content[0]["text"]

        messages = [
            system_message,

            AIMessage(
                content=summary_text
            )
        ]

        print()
        print("Підсумок:")
        print(summary_text)
        print()

    response = chain.invoke(messages)

    response_text = response.content[0]["text"]
    print(f"AI: {response_text}")
    messages.append(response)

    print()
    print("----------------------------------")
    print("HISTORY")

    for message in messages:
        print(message)

    print("----------------------------------")
    print()