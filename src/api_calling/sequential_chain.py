import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

os.environ['LANGSMITH_PROJECT'] = 'Sequential LLM App'

load_dotenv()

promt1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"]
)

promt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

model1 = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)
model2 = ChatOpenAI(model='gpt-4o', temperature=0.5)

chain = promt1 | model1 | parser | promt2 | model2 | parser

config: RunnableConfig = {
    'run_name': 'sequential_chain',
    'tags': ['llm_app', 'report_generation', 'summarisation'],
    'metadata': {
        'model1': 'gpt-4o-mini',
        'model1_temp': 0.7,
        'parser': 'StrOutputParser',
        'model2': 'gpt-4o',
        'model2_temp': 0.5
    }
}

question = input('Enter your query: ')
results = chain.invoke({"topic": question}, config=config)

print(results)