import dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool, DuckDuckGoSearchRun
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)


dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key
)

# Завдання 1
# Напишіть функцію яка перевіряє складність паролю:
#  кількість символів(>8)
#  наявність хоча б однієї літери\цифри\спеціального
# символу
#  наявність літер в різних регістрах
# Функція повертає тест з описом паролю(що добре, а що
# погано)
# На основі цієї функції створіть агента.

# @tool
# def check_password(password: str) -> str:
#     """
#     Перевіряє складність паролю.
#
#     Перевіряється:
#     - кількість символів (>8)
#     - наявність літери
#     - наявність цифри
#     - наявність спеціального символу
#     - наявність літер у різних регістрах
#     """
#
#     result = []
#
#     if len(password) > 8:
#         result.append("Добре: пароль має більше 8 символів.")
#     else:
#         result.append("Погано: пароль повинен мати більше 8 символів.")
#
#     if any(char.isalpha() for char in password):
#         result.append("Добре: пароль містить літери.")
#     else:
#         result.append("Погано: пароль не містить літер.")
#
#     if any(char.isdigit() for char in password):
#         result.append("Добре: пароль містить цифри.")
#     else:
#         result.append("Погано: пароль не містить цифр.")
#
#     if any(not char.isalnum() for char in password):
#         result.append("Добре: пароль містить спеціальний символ.")
#     else:
#         result.append("Погано: пароль не містить спеціального символу.")
#
#     has_upper = any(char.isupper() for char in password)
#     has_lower = any(char.islower() for char in password)
#
#     if has_upper and has_lower:
#         result.append("Добре: пароль містить літери у верхньому та нижньому регістрах.")
#     else:
#         result.append(
#             "Погано: пароль повинен містити літери у верхньому та нижньому регістрах."
#         )
#
#     return "\n".join(result)
#
#
# agent = create_agent(
#     model=llm,
#     tools=[check_password],
# )
#
#
# messages = [
#     SystemMessage("""
#     Ти -- ввічливий чат-бот для перевірки складності паролів.
#
#     У тебе є доступ до інструменту check_password.
#
#     ### ІНСТРУКЦІЯ ###
#     1. Коли користувач надсилає пароль, обов'язково використовуй
#        інструмент check_password для його перевірки.
#     2. Після перевірки зрозуміло повідом користувачу, що добре,
#        а що потрібно виправити.
#     3. Не вигадуй результати перевірки самостійно.
#     4. Якщо пароль слабкий, дай рекомендації щодо його покращення.
#     """)
# ]
#
# while True:
#     user_query = input("Ви: ")
#
#     if user_query == "":
#         break
#
#     messages.append(
#         HumanMessage(user_query)
#     )
#
#     data = {
#         "messages": messages
#     }
#
#     data = agent.invoke(data)
#
#     messages = data["messages"]
#
#     response = messages[-1]
#
#     print(response.text)
#
#     print()
#     print("----------ІСТОРІЯ-----------")
#
#     for message in messages:
#         print(repr(message))
#
#     print("-----------------------------")
#     print()

# Завдання 2
# Напишіть модель показує останні новини про певну
# людину. Якщо користувач вводить не ім’я людини, то вивести
# повідомлення «немає відповідної інформації»
# Скористайтесь DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

agent = create_agent(
    model=llm,
    tools=[search],
)


messages = [
    SystemMessage("""
    Ти -- ввічливий чат-бот, який показує останні новини
    про певну людину.

    У тебе є доступ до інструменту DuckDuckGo для пошуку
    інформації в інтернеті.

    ### ІНСТРУКЦІЯ ###

    1. Користувач повинен вказати ім'я людини.

    2. Якщо користувач вказав ім'я людини, використовуй
       DuckDuckGo для пошуку останніх новин про цю людину.

    3. Для пошуку використовуй запит, який містить ім'я людини
       та слова "останні новини" або "latest news".

    4. Покажи користувачу знайдену інформацію про останні новини.

    5. Якщо користувач вводить не ім'я людини, а щось інше
       (наприклад, назву міста, число, предмет або незрозумілий
       текст), не виконуй пошук.

       У такому випадку виведи:
       "немає відповідної інформації"

    6. Якщо пошук не знайшов відповідної інформації про людину,
       також виведи:
       "немає відповідної інформації"

    7. Не вигадуй новини.
    """)
]


while True:
    user_query = input("Ви: ")

    if user_query == "":
        break

    messages.append(
        HumanMessage(user_query)
    )

    data = {
        "messages": messages
    }

    data = agent.invoke(data)

    messages = data["messages"]

    response = messages[-1]

    print(response.text)

    print()
    print("----------ІСТОРІЯ-----------")

    for message in messages:
        print(repr(message))

    print("-----------------------------")
    print()
