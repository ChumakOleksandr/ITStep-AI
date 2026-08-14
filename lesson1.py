import dotenv
import os

from langchain_google_genai import GoogleGenerativeAI

dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


llm = GoogleGenerativeAI(
    model="models/gemini-3.5-flash-lite",
    api_key=api_key,
)

# Завдання 1
# Прочитайте файл data\lesson9\return_policy.txt Та
# напишіть простий чат бот для відповідей на питання
# користувачів стосовно повернення товару. Діалог завершується
# коли користувач вводить порожній рядок.
# Передавайте усю історію спілкування у форматі:
# Instruction: ….
# Human: massage1
# AI: message2
# Human: massage3
# AI: message4
# Human: massage5
# AI:

dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(
    model="models/gemini-3.5-flash-lite",
    api_key=api_key,
)

with open("data/lesson9/return_policy.txt", "r", encoding="utf-8") as file:
    return_policy = file.read()

history = []

while True:
    message = input("Human: ")

    if message == "":
        print("Діалог завершено.")
        break

    prompt = f"""
Instruction:
Ти — чат-бот магазину, який відповідає на питання користувачів
щодо повернення товарів.

Використовуй наведені нижче правила повернення товарів.
Відповідай тільки на основі цих правил. Якщо інформації
для відповіді немає, скажи, що не можеш знайти відповідь
у правилах повернення.

Правила повернення:
{return_policy}
"""

    for user_message, bot_message in history:
        prompt += f"Human: {user_message}\n"
        prompt += f"AI: {bot_message}\n"

    prompt += f"Human: {message}\n"
    prompt += "AI:"

    response = llm.invoke(prompt)

    print("AI:", response)

    history.append((message, response))