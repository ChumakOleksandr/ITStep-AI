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
# Напишіть модель для генерації персонального плану
# тренувань з двох ланцюгів:
#  Перший ланцюг отримує мету тренування(схуднення,
# набір м’язів, тощо) та повертає список вправ
#  Другий ланцюг отримує список вправ, рівень
# підготовки користувача(низький, середній,
# професіонал) та кількість часу на тиждень(в годинах)
# і повертає план тренувань

class Exercises(BaseModel):
    exercises: list[str] = Field(
        description="список вправ для заданої мети тренування"
    )


parser1 = PydanticOutputParser(
    pydantic_object=Exercises
)

instructions1 = parser1.get_format_instructions()


prompt1 = PromptTemplate.from_template("""
    Ти -- фітнес-тренер.

    Твоя задача -- підібрати список вправ відповідно
    до мети тренування користувача.

    ###ІНСТРУКЦІЇ###
    1. Враховуй мету тренування.
    2. Підбери основні вправи, які допомагають досягти цієї мети.
    3. Вкажи не більше 10 вправ.
    4. Назви тільки вправи без детального опису.

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instructions}

    ###ВХІДНІ ДАНІ###
    Мета тренування: {goal}
""",
    partial_variables={
        "format_instructions": instructions1
    }
)


chain1 = prompt1 | llm | parser1


class TrainingPlan(BaseModel):
    plan: str = Field(
        description="персональний план тренувань"
    )


parser2 = PydanticOutputParser(
    pydantic_object=TrainingPlan
)

instructions2 = parser2.get_format_instructions()


prompt2 = PromptTemplate.from_template("""
    Ти -- персональний фітнес-тренер.

    Твоя задача -- створити план тренувань
    на основі списку вправ, рівня підготовки користувача
    та кількості часу на тренування.

    ###ІНСТРУКЦІЇ###
    1. Враховуй рівень підготовки користувача.
    2. Враховуй кількість годин на тиждень.
    3. Розподіли тренування протягом тижня.
    4. Для кожного тренування вкажи вправи.
    5. Вкажи приблизну тривалість тренування.
    6. Не використовуй вправи, яких немає у списку.
    7. План повинен бути реалістичним для вказаного рівня підготовки.

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instructions}

    ###ВХІДНІ ДАНІ###

    Список вправ:
    {exercises}

    Рівень підготовки:
    {level}

    Кількість часу на тиждень:
    {hours} годин
""",
    partial_variables={
        "format_instructions": instructions2
    }
)


chain2 = prompt2 | llm | parser2


goal = "набір м'язів"

level = "середній"

hours = 5


data = {
    "goal": goal
}

response1 = chain1.invoke(data)

print("Мета тренування:", goal)

print("\nСписок вправ:")
for exercise in response1.exercises:
    print("-", exercise)


data = {
    "exercises": response1.exercises,
    "level": level,
    "hours": hours
}

response2 = chain2.invoke(data)

print("\nПлан тренувань:")
print(response2.plan)