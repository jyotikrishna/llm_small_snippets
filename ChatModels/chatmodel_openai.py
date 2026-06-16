from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

#model = ChatOpenAI(model="gpt-4")
model = ChatOpenAI(model="gpt-4", temperature=0, max_completion_tokens=10)  #temperature controls how deterministic or random/creative are, tokens are roughly words
result = model.invoke("What is the Capital of India?")

#print(result)
print(result.content)