from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

load_dotenv()

prompt = PromptTemplate.from_template("{question}")

model = ChatOpenAI()
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

question = input("Ask your question: ")
result = chain.invoke({"question":question})

print(result)