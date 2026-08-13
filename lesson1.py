# завантаження api key як змінну середовища
import dotenv
import os

from langchain_google_genai import GoogleGenerativeAI

# завантажити дані з .env
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


llm = GoogleGenerativeAI(
    model="models/gemini-3.5-flash-lite",
    api_key=api_key,
)

# response = llm.invoke("Привіт")
# print(response)

# Завдання 1
# Підключіть модель LLM за допомогою свого API key.
# Попросіть модель згенерувати:
# ● відповідь на питання у вигляді одного
# слова(наприклад яка столиця Франції?)
# ● код python
# ● коротку історію
# Підберіть параметри креативності та довжини

# response = llm.invoke(
#     "Яка столиця Франції? Відповідай лише одним словом."
# )
# print("1.", response)
#
# response = llm.invoke(
#     "Напиши Python-код для обчислення факторіала числа."
# )
# print("\n2.", response)
#
# response = llm.invoke(
#     "Напиши коротку фантастичну історію на 5 речень."
# )
# print("\n3.", response)

# Завдання 2
# Прочитайте файл data\lesson9\rules.txt з правилами
# користування атракціону. Напишіть програму яка отримує
# від користувачі питання та дає відповідь на нього виходячи
# з текстового файлу.
# Для цього об’єднайте правила користування з питанням
# користувача.
# Користувач задає питання поки не введе порожній рядок.
# Змініть файл rules.txt, щоб переконатись що модель
# дійсно його читає.

# with open("data/lesson9/rules.txt", "r", encoding="utf-8") as file:
#     rules = file.read()
#
# while True:
#     question = input("Ваше питання: ")
#
#     if question == "":
#         print("Роботу завершено.")
#         break
#
#     prompt = f"""
# Правила користування атракціоном:
#
# {rules}
#
# Відповідай на питання користувача тільки на основі цих правил.
#
# Питання:
# {question}
# """
#
#     response = llm.invoke(prompt)
#
#     print("Відповідь:", response)
#     print()

# Завдання 3
# Створіть найпростіший чат бот. Напишіть моделі якого
# персонажа вона повинна вдавати(відомий актор, персонаж
# кіно\книги, тощо).
# 2. Модель отримує інструкцію та історію попередніх
# повідомлень як від користувача, так і її власні відповіді у
# форматі
# Instruction: ….
# Human: massage1
# AI: message2
# Human: massage3
# AI: message4
# Human: massage5
# AI:

instruction = """
Ти повинен вдавати Шерлока Холмса — відомого детектива
з творів Артура Конан Дойла.

Відповідай як Шерлок Холмс:
- будь розумним та спостережливим;
- відповідай впевнено;
- використовуй логічні міркування;
- не виходь із ролі персонажа.
"""

history = []


while True:
    message = input("Human: ")

    if message.lower() == "exit":
        break

    prompt = f"Instruction: {instruction}\n"

    for user_message, bot_message in history:
        prompt += f"Human: {user_message}\n"
        prompt += f"AI: {bot_message}\n"

    prompt += f"Human: {message}\n"
    prompt += "AI:"

    response = llm.invoke(prompt)

    print("AI:", response)

    history.append((message, response))

                # Перевірка на історію
    # print("\n--- ІСТОРІЯ ДІАЛОГУ ---")
    #
    # for user_message, bot_message in history:
    #     print(f"Human: {user_message}")
    #     print(f"AI: {bot_message}")
    #
    # print("-----------------------\n")