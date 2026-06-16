from langchain_openai import OpenAI
from dotenv import load_dotenv #load secret key from .env

load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct') #Create an Object

result = llm.invoke("What is the capital of India")  #Adding prompt

print(result)

