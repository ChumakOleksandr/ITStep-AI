from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


class LearnedWords(BaseModel):
    words: list[str] = Field(
        description="список усіх англійських слів з відповіді"
    )


parser = PydanticOutputParser(
    pydantic_object=LearnedWords
)

instructions = parser.get_format_instructions()


prompt = PromptTemplate.from_template("""
    Ти -- аналізатор англійського тексту.

    Твоя задача -- знайти всі англійські слова
    у відповіді чатбота.

    ###ІНСТРУКЦІЇ###

    1. Витягни всі слова англійською мовою.
    2. Не додавай українські слова.
    3. Не додавай слова, яких немає у тексті.
    4. Не повторюй однакові слова.
    5. Поверни слова у початковій формі,
       якщо це можливо.

    ###ФОРМАТ ВІДПОВІДІ###

    {format_instructions}

    ###ТЕКСТ###

    {text}
""",
    partial_variables={
        "format_instructions": instructions
    }
)


def create_parser_chain(llm):
    return prompt | llm | parser