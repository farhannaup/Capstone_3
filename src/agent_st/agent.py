from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from qdrant_client import QdrantClient
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
import os
from typing import List


load_dotenv()

lf = Langfuse()
langfuse_handler = CallbackHandler()


def get_vector_store():
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    return QdrantVectorStore(
        client=client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embedding=embeddings,
    )

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL"),
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


class AgentInput(BaseModel):
    """
    Input schema shared by all IMDB agents
    """
    query: str = Field(description="User question about movies")
    history: str = Field(description="Summary of previous chat history")


@tool
def search_movies(query: str, k: int = 5) -> str:
    """Search movie recommendations"""
    store = get_vector_store()
    results = store.similarity_search_with_score(query, k=k)

    if not results:
        return "No movies found."

    output = []
    for i, (doc, score) in enumerate(results):
        output.append(f"""
Movie {i+1}
Title: {doc.metadata.get('title')}
Year: {doc.metadata.get('year')}
Genre: {doc.metadata.get('genre')}
Rating: {doc.metadata.get('rating')}
Score: {score:.3f}
""")
    return "\n".join(output)


@tool
def analyze_movies(query: str) -> str:
    """Analyze movie themes or story"""
    store = get_vector_store()
    docs = store.similarity_search(query, k=4)
    return "\n\n".join(d.page_content for d in docs)

@tool
def compare_movies(query: str) -> str:
    """Compare multiple movies"""
    store = get_vector_store()
    docs = store.similarity_search(query, k=6)
    return "\n\n".join(
        f"{d.metadata.get('title')} ({d.metadata.get('year')})"
        for d in docs
    )



lf_movie_search = lf.get_prompt("imdb_movie_search_agent").get_langchain_prompt()
lf_movie_analysis = lf.get_prompt("imdb_movie_analysis_agent").get_langchain_prompt()
lf_supervisor = lf.get_prompt("imdb_supervisor_agent").get_langchain_prompt()


movie_search_agent = create_agent(
    model=llm,
    tools=[search_movies],
    system_prompt=lf_movie_search,
)

movie_analysis_agent = create_agent(
    model=llm,
    tools=[analyze_movies],
    system_prompt=lf_movie_analysis,
)


def movie_search(query: str, history: str) -> str:
    """
    Use when the user wants to find or recommend movies.
    """
    result = movie_search_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query + "\nChat history:\n" + history,
                }
            ]
        },
        config={"callbacks": [langfuse_handler]},
    )
    return result["messages"][-1].text


@tool(args_schema=AgentInput)
def movie_analysis(query: str, history: str) -> str:
    """
    Use when the user asks for explanation, review, or opinion about a movie.
    """
    result = movie_analysis_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query + "\nChat history:\n" + history,
                }
            ]
        },
        config={"callbacks": [langfuse_handler]},
    )
    return result["messages"][-1].text


supervisor_agent = create_agent(
    model=llm,
    tools=[movie_search, movie_analysis, compare_movies],
    system_prompt=lf_supervisor,
)