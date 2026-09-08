import ollama
import streamlit as st
st.header('Cat Facts RAG', divider=True)
# load the dataset

dataset = []
with open('data/cat-facts.txt', 'r', encoding='utf-8') as file:
    dataset = file.readlines()
    print(f'Loaded {len(dataset)} entries')
    st.write(f'Loaded {len(dataset)} entries')


#implement vector db

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
# The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
# Each element in VECTOR_DB is a tuple (chunk, embedding).
# Streamlit re-runs this script on every interaction, so the embeddings are
# cached: they are computed once and reused for every subsequent question.
@st.cache_resource(show_spinner="Embedding the dataset...")
def build_vector_db():
  db = []
  for chunk in dataset:
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    db.append((chunk, embedding))
  return db

VECTOR_DB = build_vector_db()

#implement the retrieval function

#cosine similarity calculation
def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

#retrieval function
def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  # temporary list to store (chunk, similarity) pairs
  similarities = []
  for chunk, embedding in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  # sort by similarity in descending order, because higher similarity means more relevant chunks
  similarities.sort(key=lambda x: x[1], reverse=True)
  # finally, return the top N most relevant chunks
  return similarities[:top_n]


#Generation

#streamlit implementation

input_query = st.text_input('Ask me a question: ')
if not input_query:
  st.stop()
retrieved_knowledge = retrieve(input_query)
st.write('Retrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
  st.write(f' - (similarity: {similarity:.2f}) {chunk}')

context = '\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])

instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{context}
'''

stream = ollama.chat(
  model=LANGUAGE_MODEL,
  messages=[
    {'role': 'system', 'content': instruction_prompt},
    {'role': 'user', 'content': input_query},
  ],
  stream=True,
)

# print the response from the chatbot in real-time
st.write('Chatbot response:')
st.write_stream(chunk['message']['content'] for chunk in stream)

