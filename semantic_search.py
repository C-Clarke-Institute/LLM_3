from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from sentence_transformers.util import cos_sim

load_dotenv()

documents = [
    "Geographical Location: Sri Lanka is an island country in South Asia, located in the Indian Ocean, southeast of India. It is famous for its beautiful beaches and tropical landscapes.",
    "Capital City: Sri Jayawardenepura Kotte is the legislative capital of Sri Lanka, while Colombo is the country's main commercial and financial hub.",
    "Official Languages: Sinhala and Tamil are the official languages of Sri Lanka, while English is widely used for communication in education, business, and tourism.",
    "Rich Cultural Heritage: Sri Lanka has a history spanning thousands of years and is home to ancient cities, Buddhist temples, historic monuments, and UNESCO World Heritage Sites such as Sigiriya Rock Fortress.",
    "Economy and Natural Resources: Sri Lanka is well known for producing world-famous Ceylon tea, precious gemstones, cinnamon, and rubber. Tourism is also an important part of its economy, attracting visitors to its wildlife parks, mountains, and beaches."
]
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
doc_vectors = embedding.embed_documents(documents)
print(len(doc_vectors[0]))

def semantic_search(query, top_k=4):
    query_vec = embedding.embed_query(query)
    scores = []
    for i, doc_vector in enumerate(doc_vectors):
        score = cos_sim(query_vec, doc_vector)[0][0].item()
        scores.append((i, score)) #[(0, 0.33), (1, 0.99)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [[documents[i], s] for i,s in scores[:top_k]] #[["bla bla" , 0.99], ["adadad", "0.33"]]

print(semantic_search("What is the capital of srilanka"))
