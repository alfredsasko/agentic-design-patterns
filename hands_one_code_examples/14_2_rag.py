import os
from typing import List, TypedDict

import dotenv
import requests
import weaviate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from langgraph.graph import END, StateGraph

# Required packages:
# uv add langchain-weaviate langchain-text-splitters langchain-openai weaviate-client

# Load environment variables from .env
dotenv.load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY was not found. Add it to your .env file before running."
    )


# --- 1. Define the LangGraph state ---
class RAGGraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str


# --- 2. Load and split the source document ---
def load_documents() -> List[Document]:
    url = (
        "https://raw.githubusercontent.com/"
        "hwchase17/chroma-langchain/master/state_of_the_union.txt"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    document = Document(
        page_content=response.text,
        metadata={"source": url},
    )

    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    return text_splitter.split_documents([document])


# --- 3. Build the vector store and retriever ---
def build_retriever(
    client: weaviate.WeaviateClient,
):
    chunks = load_documents()

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )

    vectorstore = WeaviateVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 4},
    )


# --- 4. Build the LangGraph application ---
def build_rag_app(retriever):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks.

Use only the following retrieved context to answer the question.
If the answer is not present in the context, say that you do not know.
Use no more than three sentences and keep the answer concise.

Question:
{question}

Context:
{context}

Answer:
"""
    )

    rag_chain = prompt | llm | StrOutputParser()

    def retrieve_documents_node(
        state: RAGGraphState,
    ) -> RAGGraphState:
        """Retrieve documents relevant to the user's question."""
        question = state["question"]
        documents = retriever.invoke(question)

        return {
            "question": question,
            "documents": documents,
            "generation": "",
        }

    def generate_response_node(
        state: RAGGraphState,
    ) -> RAGGraphState:
        """Generate an answer using the retrieved documents."""
        question = state["question"]
        documents = state["documents"]

        context = "\n\n".join(
            document.page_content for document in documents
        )

        generation = rag_chain.invoke(
            {
                "question": question,
                "context": context,
            }
        )

        return {
            "question": question,
            "documents": documents,
            "generation": generation,
        }

    workflow = StateGraph(RAGGraphState)

    workflow.add_node(
        "retrieve",
        retrieve_documents_node,
    )
    workflow.add_node(
        "generate",
        generate_response_node,
    )

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# --- 5. Run the RAG application ---
def run_query(app, question: str) -> None:
    print(f"\nQuestion: {question}")

    initial_state: RAGGraphState = {
        "question": question,
        "documents": [],
        "generation": "",
    }

    result = app.invoke(initial_state)

    print(f"Answer: {result['generation']}")


def main() -> None:
    client = weaviate.connect_to_embedded()

    try:
        retriever = build_retriever(client)
        app = build_rag_app(retriever)

        print("\n--- Running RAG queries ---")

        run_query(
            app,
            "What did the president say about Justice Breyer?",
        )

        run_query(
            app,
            "What did the president say about the economy?",
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()