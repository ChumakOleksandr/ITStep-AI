import dotenv
import os
import json

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from parser import create_parser_chain

dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key
)

# Завдання 1
# Напишіть чат бота, який спілкується у стилі різних
# персонажів книг\фільмів або відомих людей. Ким саме бути
# чат бот вирішує з повідомлення від користувача.
# Якщо персонаж або книга невідомі, то відповісти що
# невідома інформація та запропонувати декілька відомих
# прикладів на вибір

# messages = [
#     SystemMessage("""
#     Ти -- чатбот, який може спілкуватися у стилі
#     різних персонажів книг, фільмів та відомих людей.
#
#     ###ІНСТРУКЦІЇ###
#
#     1. На початку спілкування користувач повинен вказати,
#        ким саме має бути чатбот.
#
#     2. Визнач, чи знаєш ти вказаного персонажа,
#        книгу, фільм або відому людину.
#
#     3. Якщо інформація відома, спілкуйся у відповідному
#        стилі та з урахуванням характеру цього персонажа
#        або відомої людини.
#
#     4. Не стверджуй, що ти є реальною людиною.
#        Лише імітуй стиль спілкування.
#
#     5. Якщо персонаж, книга, фільм або людина невідомі,
#        повідом користувача, що інформація невідома.
#
#     6. Якщо інформація невідома, запропонуй кілька
#        відомих варіантів на вибір.
#
#     Наприклад:
#     - Гаррі Поттер
#     - Шерлок Холмс
#     - Тоні Старк
#     - Альбус Дамблдор
#
#     7. Якщо користувач вибрав відомого персонажа,
#        продовжуй спілкування в його стилі.
#
#     8. Відповіді повинні бути короткими та зрозумілими.
#
#     9. Спілкуйся тією ж мовою, якою пише користувач.
#     """)
# ]
#
#
# trimmer = trim_messages(
#     strategy="last",
#
#     token_counter=len,
#
#     max_tokens=10,
#
#     start_on="human",
#
#     end_on="human",
#
#     include_system=True
# )
#
#
# chain = trimmer | llm
#
#
# while True:
#     user_text = input("Ви: ")
#
#     if user_text == "":
#         break
#
#     human_message = HumanMessage(
#         content=user_text
#     )
#
#     messages.append(human_message)
#
#     response = chain.invoke(messages)
#
#     print(f"AI: {response.content[0]['text']}")
#
#     messages.append(response)
#
#
#     print()
#     print("----------------------------------")
#     print("HISTORY")
#
#     for message in messages:
#         print(message)
#
#     print("----------------------------------")
#     print()
#
# Завдання 2
# Напишіть чат бота, який дає відповіді на питання
# стосовно умов повернення товару.
# Якщо користувач запитує щось інше, то відповідати що
# немає інформації.
# Застосуйте обмеження історії(можна десь 5 повідомлень)

# with open(
#     "data/lesson9/return_policy.txt",
#     "r",
#     encoding="utf-8"
# ) as file:
#     return_policy = file.read()
#
#
# messages = [
#     SystemMessage(f"""
#     Ти -- чатбот магазину.
#
#     Твоя задача -- відповідати на питання користувача
#     стосовно умов повернення товару.
#
#     ###ІНСТРУКЦІЇ###
#
#     1. Використовуй тільки інформацію з правил повернення,
#        які наведені нижче.
#
#     2. Якщо питання стосується повернення товару,
#        дай коротку та зрозумілу відповідь.
#
#     3. Якщо у правилах немає відповіді на питання,
#        скажи, що інформації щодо цього питання немає.
#
#     4. Якщо користувач запитує щось, що не стосується
#        повернення товару, відповідай:
#        "На жаль, у мене немає інформації щодо цього питання."
#
#     5. Не вигадуй інформацію.
#
#     6. Відповіді повинні бути короткими.
#
#     ###ПРАВИЛА ПОВЕРНЕННЯ ТОВАРУ###
#
#     {return_policy}
#     """)
# ]
#
#
# trimmer = trim_messages(
#     strategy="last",
#
#     token_counter=len,
#
#     max_tokens=5,
#
#     start_on="human",
#
#     end_on="human",
#
#     include_system=True
# )
#
#
# chain = trimmer | llm
#
#
# while True:
#     user_text = input("Ви: ")
#
#     if user_text == "":
#         break
#
#     human_message = HumanMessage(
#         content=user_text
#     )
#
#     messages.append(human_message)
#
#     response = chain.invoke(messages)
#
#     print(f"AI: {response.content[0]['text']}")
#
#     messages.append(response)
#
#
#     print()
#     print("----------------------------------")
#     print("HISTORY")
#
#     for message in messages:
#         print(message)
#
#     print("----------------------------------")
#     print()

# Завдання 3
# Напишіть чат бота, який допомагає у вивченні
# англійської мови з наступним функціоналом:
#  якщо користувач просить перекласти слово або фразу
# то дається переклад слова та приклад використання в
# реченні
#  якщо користувач просить перекласти речення, то
# Практичне завдання
# дається переклад самого речення, а також пояснення
# граматики, наприклад структура there is\are, питання в
# різних часових формах, тощо.
# Приклади реалізуйте як HumanMessage та AIMessage

# messages = [
#     SystemMessage("""
#     Ти -- помічник для вивчення англійської мови.
#
#     Твоя задача допомагати користувачу перекладати
#     англійські слова, фрази та речення.
#
#     ###ІНСТРУКЦІЇ###
#
#     1. Якщо користувач просить перекласти слово або коротку
#        фразу, дай:
#        - переклад;
#        - приклад використання цього слова або фрази
#          в англійському реченні;
#        - переклад прикладу українською мовою.
#
#     2. Якщо користувач просить перекласти ціле речення,
#        дай:
#        - переклад речення;
#        - пояснення граматики;
#        - поясни структуру речення;
#        - якщо використовується певний час, поясни його.
#
#     3. Пояснюй граматику просто та зрозуміло.
#
#     4. Якщо в реченні використовується конструкція
#        There is / There are, поясни її використання.
#
#     5. Якщо речення є питанням, поясни структуру питання
#        та порядок слів.
#
#     6. Якщо використовується певний граматичний час,
#        поясни його структуру та використання.
#
#     7. Відповіді повинні бути зрозумілими для людини,
#        яка вивчає англійську мову.
#
#     8. Спілкуйся тією ж мовою, якою користувач ставить питання.
#     """),
#
#
#     HumanMessage("""
#     Переклади слово "beautiful"
#     """),
#
#     AIMessage("""
#     Переклад: красивий, прекрасний.
#
#     Приклад:
#     She has a beautiful dress.
#
#     Переклад прикладу:
#     У неї красива сукня.
#     """),
#
#
#     HumanMessage("""
#     Переклади речення "There are three books on the table."
#     """),
#
#     AIMessage("""
#     Переклад:
#     На столі є три книги.
#
#     Граматика:
#     There are використовується, коли ми говоримо
#     про наявність декількох предметів у певному місці.
#
#     There are + множина:
#     There are three books.
#
#     There is використовується з одним предметом:
#     There is a book on the table.
#
#     У цьому реченні "three books" -- множина,
#     тому використовується "There are".
#     """)
# ]
#
#
# trimmer = trim_messages(
#     strategy="last",
#
#     token_counter=len,
#
#     max_tokens=7,
#
#     start_on="human",
#
#     end_on="human",
#
#     include_system=True
# )
#
#
# chain = trimmer | llm
#
# while True:
#     user_text = input("Ви: ")
#
#     if user_text == "":
#         break
#
#     human_message = HumanMessage(
#         content=user_text
#     )
#
#     messages.append(human_message)
#
#     response = chain.invoke(messages)
#
#     print(f"AI: {response.content[0]['text']}")
#
#     messages.append(response)
#
#
#     print()
#     print("----------------------------------")
#     print("HISTORY")
#
#     for message in messages:
#         print(message)
#
#     print("----------------------------------")
#     print()

# Завдання 4
# Модифікуйте попереднє завдання таким чином, щоб в
# SystemMessage передавався список вивчених слів
# користувачем.
# Для цього напишіть окрему модель яка буде діставати з
# відповіді(AIMessage) усі англійські слова(вважаємо що
# користувач знає лише ті слова, про які йому сказала модель).
# Список вивчених слів треба зберігати в json файлі та
# відвантажувати при запуску програми.
# Змініть функціонал таким чином:
#  якщо користувач просить перекласти слово або фразу
# то дається переклад слова та приклад використання в
# реченні з вивченими словами
#  якщо користувач просить перекласти речення, то
# додатково пояснюється значення невідомих слів

file_name = "learned_words.json"


if os.path.exists(file_name):

    with open(
        file_name,
        "r",
        encoding="utf-8"
    ) as file:

        learned_words = json.load(file)

else:

    learned_words = []


print("Вивчені слова:")
print(learned_words)



words_chain = create_parser_chain(llm)


messages = [
    SystemMessage(f"""
    Ти -- помічник для вивчення англійської мови.

    Твоя задача допомагати користувачу перекладати
    англійські слова, фрази та речення.

    ###ВИВЧЕНІ СЛОВА КОРИСТУВАЧА###

    {learned_words}

    Вважається, що користувач знає тільки ті англійські
    слова, які є у цьому списку.

    ###ІНСТРУКЦІЇ###

    1. Якщо користувач просить перекласти слово або коротку
       фразу:

       - дай переклад;
       - дай приклад використання в реченні;
       - у прикладі використовуй якомога більше слів
         зі списку вивчених слів;
       - дай переклад прикладу українською мовою.

    2. Якщо користувач просить перекласти ціле речення:

       - дай переклад речення;
       - поясни граматику;
       - визнач невідомі слова;
       - поясни значення кожного невідомого слова.

    3. Невідомими вважаються слова,
       яких немає у списку вивчених слів.

    4. Якщо використовується конструкція
       There is / There are -- поясни її.

    5. Якщо речення є питанням -- поясни порядок слів.

    6. Якщо використовується певний час -- поясни
       його структуру та використання.

    7. Не вигадуй інформацію.

    8. Відповіді повинні бути зрозумілими.

    9. Спілкуйся тією ж мовою, якою користувач
       ставить питання.
    """)
]



messages.append(
    HumanMessage("""
    Переклади слово "beautiful"
    """)
)

messages.append(
    AIMessage("""
    Переклад: красивий, прекрасний.

    Приклад:
    She has a beautiful dress.

    Переклад:
    У неї красива сукня.
    """)
)

messages.append(
    HumanMessage("""
    Переклади речення "There are three books on the table."
    """)
)

messages.append(
    AIMessage("""
    Переклад:
    На столі є три книги.

    Граматика:
    There are використовується, коли ми говоримо
    про наявність декількох предметів.

    There are + множина:
    There are three books.

    There is використовується з одним предметом:
    There is a book.

    Невідоме слово:
    table -- стіл.
    """)
)


trimmer = trim_messages(
    strategy="last",

    token_counter=len,

    max_tokens=7,

    start_on="human",

    end_on="human",

    include_system=True
)


chain = trimmer | llm

while True:

    user_text = input("Ви: ")

    if user_text == "":
        break


    human_message = HumanMessage(
        content=user_text
    )

    messages.append(human_message)


    response = chain.invoke(messages)

    response_text = response.content[0]["text"]

    print()
    print("AI:", response_text)


    messages.append(response)


    data = {
        "text": response_text
    }

    words_response = words_chain.invoke(data)



    for word in words_response.words:

        if word.lower() not in [
            w.lower() for w in learned_words
        ]:

            learned_words.append(word)


    with open(
        file_name,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            learned_words,
            file,
            ensure_ascii=False,
            indent=4
        )


    print()
    print("----------------------------------")
    print("HISTORY")

    for message in messages:
        print(message)

    print("----------------------------------")

    print()
    print("Вивчені слова:")
    print(learned_words)

    print()