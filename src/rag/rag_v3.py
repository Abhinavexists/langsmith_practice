import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from langsmith import traceable
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableConfig,
)
from langchain_community.vectorstores import FAISS

os.environ["LANGSMITH_PROJECT"] = "rag_bot_v3"
load_dotenv()

PDF_PATH = "data/stats.pdf"
INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)

# ----------------------------
# Core pipeline steps
# ----------------------------

@traceable(name="load_pdf", tags=["load_pdf"])
def load_pdf(pdf_path: str):
    return PyMuPDFLoader(pdf_path).load()

@traceable(name="splitter", tags=["splitting"])
def splitter(docs, chunk_size: int = 1000, chunk_overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="embeddings", tags=["embeddings"])
def embeddings(splits, embed_model_name: str):
    emb = OpenAIEmbeddings(model=embed_model_name)
    return FAISS.from_documents(splits, emb)

def file_fingerprint(file_path: str) -> dict:
    p = Path(file_path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {
        "sha256": h.hexdigest(),
        "size": p.stat().st_size,
        "mtime": int(p.stat().st_mtime),
    }

def index_key(pdf_path: str, chunk_size: int, chunk_overlap: int, embed_model_name: str):
    meta = {
        "pdf_fingerprint": file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
        "format": "v1",
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()

@traceable(name="load_index", tags=["index"])
def load_index(index_dir: Path, embed_model_name: str, pdf_path: str, chunk_size: int, chunk_overlap: int):
    emb = OpenAIEmbeddings(model=embed_model_name)
    try:
        return FAISS.load_local(
            str(index_dir), emb, allow_dangerous_deserialization=True
        )
    except Exception:
        print("Index corrupted. Rebuilding...")
        return build_index(pdf_path, index_dir, chunk_size, chunk_overlap, embed_model_name)

@traceable(name="build_index", tags=["index"])
def build_index(pdf_path: str, index_dir: Path, chunk_size: int, chunk_overlap: int, embed_model_name: str):
    docs = load_pdf(pdf_path)
    splits = splitter(docs, chunk_size, chunk_overlap)
    vector_store = embeddings(splits, embed_model_name)
    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_dir))

    (index_dir / "meta.json").write_text(
        json.dumps(
            {
                "PDF_PATH": os.path.abspath(pdf_path),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embed_model": embed_model_name,
            },
            indent=2,
        )
    )
    return vector_store

def dispatcher(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = "text-embedding-3-small",
    force_rebuild: bool = False,
):
    key = index_key(pdf_path, chunk_size, chunk_overlap, embed_model_name)
    index_dir = INDEX_ROOT / key
    use_cache = index_dir.exists() and not force_rebuild

    if use_cache:
        return load_index(index_dir, embed_model_name, pdf_path, chunk_size, chunk_overlap)
    else:
        return build_index(pdf_path, index_dir, chunk_size, chunk_overlap, embed_model_name)

# ----------------------------
# LLM + Retrieval pipeline
# ----------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# ----------------------------
# Main Entry
# ----------------------------

if __name__ == "__main__":
    vectorstore = dispatcher(PDF_PATH)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    parallel = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    chain = parallel | prompt | llm | StrOutputParser()

    print("PRINT RAG - Ask a question")
    q = input("\nQ: ").strip()

    config: RunnableConfig = {"run_name": "rag_bot_v3"}

    ans = chain.invoke(q, config=config)
    print("\nA:", ans)
