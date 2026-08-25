import os
import dotenv

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key
)
# Завдання 1
# Напишіть модель для рекомендації книг з двох ланцюгів:
#  Перший ланцюг отримує назву книги та визначає її
# жанр
#  Другий отримує назву книги, жанр та повертає список
# схожих книг(того ж самого жанру та іншого)

# class BookGenre(BaseModel):
#     genre: str = Field(description="жанр книги")
#
#
# parser1 = PydanticOutputParser(
#     pydantic_object=BookGenre
# )
#
# instructions1 = parser1.get_format_instructions()
#
# prompt1 = PromptTemplate.from_template("""
#     Ти -- літературний експерт.
#     Твоя задача визначити жанр книги за її назвою.
#
#     ###ІНСТРУКЦІЇ###
#     1. Визнач основний жанр книги.
#     2. Якщо книга належить до декількох жанрів,
#        вкажи основний жанр.
#
#     ###ФОРМАТ ВІДПОВІДІ###
#     {format_instructions}
#
#     ###ВХІДНІ ДАНІ###
#     Назва книги: {book_name}
# """,
#     partial_variables={
#         "format_instructions": instructions1
#     }
# )
#
# chain1 = prompt1 | llm | parser1
#
#
#
# class Recommendations(BaseModel):
#     books: list[str] = Field(
#         description="список схожих книг"
#     )
#
#
# parser2 = PydanticOutputParser(
#     pydantic_object=Recommendations
# )
#
# instructions2 = parser2.get_format_instructions()
#
# prompt2 = PromptTemplate.from_template("""
#     Ти -- консультант книжкового магазину.
#
#     Твоя задача рекомендувати книги, схожі на книгу,
#     яку вказав користувач.
#
#     ###ІНСТРУКЦІЇ###
#     1. Рекомендуй не більше 5 книг.
#     2. Частина книг повинна бути того ж жанру.
#     3. Також можеш рекомендувати книги інших жанрів,
#        якщо вони схожі за тематикою, сюжетом або стилем.
#     4. Не рекомендуй книгу, яку вказав користувач.
#     5. Вказуй тільки назви книг.
#
#     ###ФОРМАТ ВІДПОВІДІ###
#     {format_instructions}
#
#     ###ВХІДНІ ДАНІ###
#     Назва книги: {book_name}
#     Жанр книги: {genre}
# """,
#     partial_variables={
#         "format_instructions": instructions2
#     }
# )
#
# chain2 = prompt2 | llm | parser2
#
#
# user_book = "Гаррі Поттер і філософський камінь"
#
# data = {
#     "book_name": user_book
# }
#
# response1 = chain1.invoke(data)
#
# print(f"Книга: {user_book}")
# print(f"Жанр: {response1.genre}")
#
# data = {
#     "book_name": user_book,
#     "genre": response1.genre
# }
#
# response2 = chain2.invoke(data)
#
# print("Схожі книги:")
# for book in response2.books:
#     print(book)

# Завдання 2
# Напишіть модель для генерації листа:
#  Перший ланцюг отримує короткий опис листа та
# генерує основний зміст
#  Другий ланцюг отримує основний зміст та стиль
# листа(формальний, неформальний, тощо) та генерує
# лист

# class LetterContent(BaseModel):
#     content: str = Field(
#         description="основний зміст листа"
#     )
#
#
# parser1 = PydanticOutputParser(
#     pydantic_object=LetterContent
# )
#
# instructions1 = parser1.get_format_instructions()
#
#
# prompt1 = PromptTemplate.from_template("""
#     Ти -- помічник для написання листів.
#
#     Твоя задача -- за коротким описом листа
#     створити його основний зміст.
#
#     ###ІНСТРУКЦІЇ###
#     1. Передай всю основну інформацію з опису.
#     2. Не додавай інформацію, якої немає в описі.
#     3. Текст повинен бути логічним та зрозумілим.
#     4. Не додавай привітання та підпис,
#        оскільки вони будуть додані на наступному етапі.
#
#     ###ФОРМАТ ВІДПОВІДІ###
#     {format_instructions}
#
#     ###ВХІДНІ ДАНІ###
#     Опис листа: {description}
# """,
#     partial_variables={
#         "format_instructions": instructions1
#     }
# )
#
#
# chain1 = prompt1 | llm | parser1
#
#
# class Letter(BaseModel):
#     letter: str = Field(
#         description="готовий лист"
#     )
#
#
# parser2 = PydanticOutputParser(
#     pydantic_object=Letter
# )
#
# instructions2 = parser2.get_format_instructions()
#
#
# prompt2 = PromptTemplate.from_template("""
#     Ти -- професійний автор листів.
#
#     Твоя задача -- перетворити основний зміст
#     у готовий лист відповідно до заданого стилю.
#
#     ###ІНСТРУКЦІЇ###
#     1. Дотримуйся вказаного стилю.
#     2. Формальний стиль -- ввічливий та офіційний.
#     3. Неформальний стиль -- дружній та простий.
#     4. Не змінюй основний зміст.
#     5. Додай відповідне привітання та завершення листа.
#
#     ###ФОРМАТ ВІДПОВІДІ###
#     {format_instructions}
#
#     ###ВХІДНІ ДАНІ###
#     Основний зміст:
#     {content}
#
#     Стиль листа:
#     {style}
# """,
#     partial_variables={
#         "format_instructions": instructions2
#     }
# )
#
#
# chain2 = prompt2 | llm | parser2
#
#
# description = """
# Потрібно написати лист викладачу з проханням
# перенести термін здачі завдання на наступний тиждень,
# тому що я не встигаю завершити роботу.
# """
#
# style = "формальний"
#
#
# data = {
#     "description": description
# }
#
#
# response1 = chain1.invoke(data)
#
# print("Основний зміст:")
# print(response1.content)
#
#
# data = {
#     "content": response1.content,
#     "style": style
# }
#
#
# response2 = chain2.invoke(data)
#
# print("\nГотовий лист:")
# print(response2.letter)

# Завдання 3
# Напишіть модель для генерації резюме:
#  Перший ланцюг отримує опис вакансії та повертає
# основні навички, які необхідні
#  Другий ланцюг отримує основні навички та опис
# кандидата і генерує резюме
# структура відповіді

class JobSkills(BaseModel):
    skills: list[str] = Field(
        description="список основних навичок, необхідних для вакансії"
    )


parser1 = PydanticOutputParser(
    pydantic_object=JobSkills
)

instructions1 = parser1.get_format_instructions()


prompt1 = PromptTemplate.from_template("""
    Ти -- спеціаліст з аналізу вакансій.

    Твоя задача -- проаналізувати опис вакансії
    та визначити основні навички, які необхідні кандидату.

    ###ІНСТРУКЦІЇ###
    1. Визнач тільки найважливіші навички.
    2. Враховуй технічні та професійні навички.
    3. Не додавай навички, яких немає в описі вакансії.
    4. Вкажи не більше 10 навичок.

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instructions}

    ###ОПИС ВАКАНСІЇ###
    {job_description}
""",
    partial_variables={
        "format_instructions": instructions1
    }
)


chain1 = prompt1 | llm | parser1



class Resume(BaseModel):
    resume: str = Field(
        description="готове резюме кандидата"
    )


parser2 = PydanticOutputParser(
    pydantic_object=Resume
)

instructions2 = parser2.get_format_instructions()


prompt2 = PromptTemplate.from_template("""
    Ти -- професійний спеціаліст з написання резюме.

    Твоя задача -- створити резюме кандидата
    на основі опису кандидата та навичок,
    необхідних для вакансії.

    ###ІНСТРУКЦІЇ###
    1. Використовуй інформацію з опису кандидата.
    2. Виділи навички кандидата, які відповідають вакансії.
    3. Не вигадуй досвід або навички, яких немає в описі кандидата.
    4. Резюме повинно бути професійним та структурованим.
    5. Адаптуй резюме під дану вакансію.

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instructions}

    ###НЕОБХІДНІ НАВИЧКИ###
    {skills}

    ###ОПИС КАНДИДАТА###
    {candidate}
""",
    partial_variables={
        "format_instructions": instructions2
    }
)


chain2 = prompt2 | llm | parser2


job_description = """
Потрібен Python розробник для роботи над computer vision проектами.

Вимоги:
- Python
- OpenCV
- PyTorch
- YOLO
- Git
- досвід роботи з computer vision
- розуміння алгоритмів машинного навчання
"""


candidate = """
Олександр, Python розробник.

Маю досвід програмування на Python.
Працював з OpenCV та YOLO для задач комп'ютерного зору.
Маю базові знання PyTorch та машинного навчання.
Використовую Git для роботи з кодом.
Розробляв проекти з детекції об'єктів.
"""


data = {
    "job_description": job_description
}

response1 = chain1.invoke(data)

print("Необхідні навички:")
for skill in response1.skills:
    print("-", skill)


data = {
    "skills": response1.skills,
    "candidate": candidate
}

response2 = chain2.invoke(data)

print("\nРезюме:")
print(response2.resume)