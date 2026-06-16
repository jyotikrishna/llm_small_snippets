from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as NotImplemented


load_dotenv()

embedding = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions=300)

documents = [                                                                                   #Documents 
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about virat kohli'         #User query

#Create the documents and user query
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

#Find the similarity scores
sim_scores = cosine_similarity([query_embedding], doc_embeddings)[0]


index, score = sorted(list(enumerate(sim_scores)), key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("Similarity score is:", score)


#Usually the documents should be saved in a database called as vector database.
#I could not run this code because I have not paid for tokens.

