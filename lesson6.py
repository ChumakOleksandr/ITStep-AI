import dotenv
import os
import json
import uuid

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from pinecone import Pinecone
from pinecone import ServerlessSpec

dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key
)

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=api_key,
)

pc = Pinecone(
    api_key=pinecone_api_key
)

# Завдання 1
# Створіть векторну базу даних, де кожен документ – це
# вміст файлу з папки data/lesson_rag/files
#  добавте в метадані шлях до файлу
#  створіть для кожного документу ID
#  збережіть створені ID та назви відповідних файлів в
# окремий json файл
# Перевірте чи працює правильно пошук

# index_name = "lesson-rag"
#
#
# if not pc.has_index(index_name):
#     pc.create_index(
#         name=index_name,
#         dimension=3072,
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )
#
#
# index = pc.Index(index_name)
#
#
# vector_store = PineconeVectorStore(
#     index=index,
#     embedding=embedding
# )
#
# folder_path = "data/lesson_rag/files"
#
# documents = []
# id_file_list = []
#
#
# for file_name in os.listdir(folder_path):
#
#     file_path = os.path.join(
#         folder_path,
#         file_name
#     )
#
#     if not os.path.isfile(file_path):
#         continue
#
#     with open(
#         file_path,
#         "r",
#         encoding="utf-8"
#     ) as f:
#         text = f.read()
#
#     document_id = str(uuid.uuid4())
#
#     document = Document(
#         page_content=text,
#         metadata={
#             "file_path": file_path,
#             "file_name": file_name,
#             "id": document_id
#         }
#     )
#
#     documents.append(document)
#
#     id_file_list.append({
#         "id": document_id,
#         "file_name": file_name
#     })
#
#
# ids = [
#     document.metadata["id"]
#     for document in documents
# ]
#
#
# vector_store.add_documents(
#     documents=documents,
#     ids=ids
# )
#
#
# print("Документи додані у векторну базу даних.")
# print("Кількість документів:", len(documents))
#
#
# with open(
#     "document_ids.json",
#     "w",
#     encoding="utf-8"
# ) as f:
#
#     json.dump(
#         id_file_list,
#         f,
#         ensure_ascii=False,
#         indent=4
#     )
#
#
# print("ID та назви файлів збережені у document_ids.json")
#
#
# query = input(
#     "\nВведіть запит для пошуку: "
# )
#
#
# results = vector_store.similarity_search(
#     query,
#     k=3
# )
#
#
# print()
# print("========== РЕЗУЛЬТАТ ПОШУКУ ==========")
#
#
# for i, result in enumerate(results):
#
#     print()
#     print(f"Документ №{i + 1}")
#
#     print("ID:", result.metadata.get("id"))
#
#     print(
#         "Файл:",
#         result.metadata.get("file_name")
#     )
#
#     print(
#         "Шлях:",
#         result.metadata.get("file_path")
#     )
#
#     print(
#         "Вміст:",
#         result.page_content[:500]
#     )
#
#     print("--------------------------------------")

# Завдання 2
# На основі створеної бази даних створіть агента та
# реалізуйте його у вигляді чат бота

pc = Pinecone(
    api_key=pinecone_api_key
)

index_name = "lesson-rag"

index = pc.Index(index_name)


vector_store = PineconeVectorStore(
    index=index,
    embedding=embedding
)


@tool
def document_search(query: str) -> str:
    """
    Шукає інформацію у векторній базі даних.

    Використовуй цей інструмент, якщо користувач
    запитує інформацію, яка може знаходитися
    у документах.
    """

    results = vector_store.similarity_search(
        query,
        k=3
    )

    if not results:
        return "Інформацію не знайдено."

    result_text = []

    for document in results:

        file_name = document.metadata.get(
            "file_name",
            "невідомий файл"
        )

        file_path = document.metadata.get(
            "file_path",
            "невідомий шлях"
        )

        result_text.append(
            f"Файл: {file_name}\n"
            f"Шлях: {file_path}\n"
            f"Вміст:\n{document.page_content}"
        )

    return "\n\n--------------------\n\n".join(result_text)


agent = create_agent(
    model=llm,
    tools=[document_search]
)


messages = [
    SystemMessage("""
    Ти -- ввічливий чат-бот, який відповідає
    на питання користувача на основі документів
    з векторної бази даних.

    ### ІНСТРУКЦІЯ ###

    1. Якщо користувач питає щось, що може бути
       у документах, обов'язково використовуй
       інструмент document_search.

    2. Відповідай на основі знайденої інформації.

    3. Не вигадуй інформацію, якої немає
       у документах.

    4. Якщо необхідної інформації у документах
       немає, чесно повідом про це.

    5. Відповідай коротко та зрозуміло.
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

    print()
    print("----------ІСТОРІЯ-----------")

    for message in messages:
        print(repr(message))

    print("-----------------------------")
    print()

