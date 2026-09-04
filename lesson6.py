
import dotenv
import os
import json
import uuid

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain.agents import create_agent
from langchain_core.tools import tool

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from langchain_core.documents import Document

from pinecone import Pinecone

dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key
)

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=api_key
)

pc = Pinecone(
    api_key=pinecone_api_key
)

# Завдання 1
# Добавте в створену базу даних файл
# data/lesson_rag/huge_file.txt про умови користування гуглом
# Оскільки файл надто великий, то його треба добавляти
# частинами. Для цього:
#  прочитайте вміст файлу
#  розділіть його на окремі блоки(між блоками два
# порожніх рядка, дивись файл)
#  отримайте перший рядок кожного блоку – це його
# назва
#  створіть документи для кожного блоку. В метаданих:
# o назва файлу
# o назва блоку
#  створіть ID та добавте все в існуючу базу даних
#  добавте ID у json файл
#  перевірте агента

index_name = "lesson-rag"

index = pc.Index(index_name)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embedding
)

file_path = "data/lesson_rag/huge_file.txt"
file_name = "huge_file.txt"

with open(
    file_path,
    "r",
    encoding="utf-8"
) as f:
    text = f.read()

blocks = text.split("\n\n\n")


documents = []
new_ids = []


for block in blocks:
    block = block.strip()

    if not block:
        continue

    lines = block.splitlines()

    block_name = lines[0].strip()

    document_id = str(uuid.uuid4())

    document = Document(
        page_content=block,
        metadata={
            "file_name": file_name,
            "block_name": block_name
        }
    )

    documents.append(document)
    new_ids.append(document_id)

if documents:

    vector_store.add_documents(
        documents=documents,
        ids=new_ids
    )


print(
    f"Додано блоків: {len(documents)}"
)

json_file = "document_ids.json"

if os.path.exists(json_file):

    with open(
        json_file,
        "r",
        encoding="utf-8"
    ) as f:

        id_file_list = json.load(f)

else:
    id_file_list = []

for document_id, document in zip(
    new_ids,
    documents
):

    id_file_list.append({
        "id": document_id,
        "file_name": document.metadata["file_name"],
        "block_name": document.metadata["block_name"]
    })

with open(
    json_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        id_file_list,
        f,
        ensure_ascii=False,
        indent=4
    )


print(
    f"JSON оновлено: {json_file}"
)

@tool
def document_search(query: str) -> str:
    """
    Шукає інформацію у векторній базі даних.
    """

    results = vector_store.similarity_search(
        query,
        k=3
    )

    if not results:
        return "Інформацію не знайдено."

    result = []

    for document in results:

        result.append(
            f"Файл: {document.metadata.get('file_name')}\n"
            f"Блок: {document.metadata.get('block_name')}\n"
            f"Вміст:\n{document.page_content}"
        )

    return "\n\n--------------------\n\n".join(result)


agent = create_agent(
    model=llm,
    tools=[document_search]
)

messages = [
    SystemMessage("""
    Ти -- ввічливий чат-бот, який відповідає
    на питання на основі документів
    з векторної бази даних.

    Якщо питання стосується інформації
    з документів, обов'язково використовуй
    document_search.

    Не вигадуй інформацію.

    Відповідай тільки на основі знайдених документів.
    """)
]

# Перевірка
while True:
    user_query = input("\nВи: ")

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

    print("\nВідповідь:")
    print(response.text)
