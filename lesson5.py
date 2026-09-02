import dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)


dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
serper_key = os.getenv("SERPER_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key
)

# Завдання 1
# Напишіть чат бота, з інструментом по рекомендації
# ресторанів.
# Для цього скористайтесь
# GoogleSerperAPIWrapper(type="places")
# Інструмент повинен отримувати запит для пошуку та
# повертати таку інформацію про ресторани:
#  назва
#  посилання на сайт(якщо є)
#  рейтинг

serper_search = GoogleSerperAPIWrapper(
    serper_api_key=serper_key,
    type="places"
)

@tool
def find_restaurants(query: str) -> str:
    """
    Шукає ресторани за запитом.

    Повертає:
    - назву ресторану
    - посилання на сайт
    - рейтинг
    """

    print("hi from find_restaurants tool")

    result = serper_search.results(query)

    restaurants = []

    for place in result.get("places", []):
        name = place.get("title", "Немає назви")
        rating = place.get("rating", "Немає рейтингу")
        website = place.get("website", "Немає посилання")

        restaurants.append(
            f"Назва: {name}\n"
            f"Сайт: {website}\n"
            f"Рейтинг: {rating}"
        )

    if not restaurants:
        return "Ресторанів за цим запитом не знайдено."

    return "\n\n".join(restaurants)


agent = create_agent(
    model=llm,
    tools=[find_restaurants]
)

messages = [
    SystemMessage("""
    Ти -- ввічливий чат-бот для рекомендації ресторанів.

    У тебе є інструмент find_restaurants.

    ### ІНСТРУКЦІЯ ###

    1. Якщо користувач просить порекомендувати ресторани,
       обов'язково використовуй інструмент find_restaurants.

    2. Передай у інструмент запит користувача.

    3. Для кожного знайденого ресторану покажи:
       - назву
       - посилання на сайт, якщо воно є
       - рейтинг

    4. Не вигадуй ресторани та їхні рейтинги.

    5. Якщо сайт відсутній, напиши:
       "Сайт відсутній".

    6. Відповідай коротко та зрозуміло.
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

    print()
    print(response.text)

    # виведення історії
    print()
    print("----------ІСТОРІЯ-----------")

    for message in messages:
        print(repr(message))

    print("-----------------------------")
    print()

