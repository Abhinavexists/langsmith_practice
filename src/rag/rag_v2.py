# tracing the problem  
import os
from dotenv import load_dotenv
from langsmith import traceable
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain_community.vectorstores import FAISS

os.environ['LANGSMITH_PROJECT'] = 'rag_bot_v2'

load_dotenv()

PDF_PATH = "data/stats.pdf"

@traceable(name='loader', tags=['pdfLoader'], metadata={'pdf_model':'PyMuPDFLoader'})
def loader(PDF_PATH: str):
    loader = PyMuPDFLoader(PDF_PATH)
    return loader.load()

@traceable(name='splitter', tags=['splitting'], metadata={'splitting_method':'RecusiveCharacterTextSplitter'})
def splitter(docs, chunk_size = 1000, chunk_overlap = 150):
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    return splitter.split_documents(docs)

@traceable(name='embeddings', tags=['embeddings','vector_store'], metadata={'embedding_model':'text-embedding-small-3'})
def embeddings(splits):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(splits, embeddings)

@traceable(name='retriever', tags=['retriever'])
def retriever(vector_store, search_type='similarity'):
    return vector_store.as_retriever(search_type=search_type, search_kwargs={"k": 4})

@traceable(name='pipeline', tags=['setup'])
def pipeline(PDF_PATH: str):
    docs = loader(PDF_PATH)
    split = splitter(docs)
    vector_store = embeddings(split)
    return retriever(vector_store)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

retrieve = pipeline(PDF_PATH)

parallel = RunnableParallel({
    "context": retrieve | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

print("PRINT RAG - Ask a question")
q = input("\nQ: ").strip()

config: RunnableConfig = {
    'run_name': 'rag_bot_v2',
}

ans = chain.invoke(q, config=config)
print("\nA:", ans)