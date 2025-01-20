import streamlit as st
import ollama
import pickle

# Define your constants for models
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Load vector database (VECTOR_DB) from a pickle file
VECTOR_DB = []

with open("vector_db.pkl", "rb") as f:
    VECTOR_DB = pickle.load(f)

# Cosine similarity function to compare the query and embeddings
def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

# Function to retrieve relevant chunks based on cosine similarity
def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Streamlit User Interface
st.title("AI Chatbot with Knowledge Retrieval")

st.markdown("""
    <style>
        .stTextInput > div > div > input {
            border: 2px solid #00bcd4;  /* Cyan border color */
            border-radius: 5px;  /* Optional: rounded corners */
            padding: 10px;  /* Optional: space inside the box */
        }
    </style>
""", unsafe_allow_html=True)
# Add a text input for the user's query
input_query = st.text_input("Ask me a question:")

if input_query:
    # Retrieve the most relevant chunks based on the query
    retrieved_knowledge = retrieve(input_query)

    # Display retrieved knowledge (top N chunks)
    st.subheader("Retrieved knowledge:")
    for chunk, similarity in retrieved_knowledge:
        st.write(f"(Similarity: {similarity:.2f}) {chunk}")

    # Construct the instruction prompt for the chatbot
    backslash_char = "\\"
    instruction_prompt = f'''You are a helpful chatbot.
    Use only the following pieces of context to answer the question. Don't make up any new information:
    {'{backslash_char}n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
    '''

    # Interact with the chatbot model
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=True,
    )

    # Display chatbot response in real-time
    st.subheader("Chatbot response:")
    chatbot_response = ""
    for chunk in stream:
        chatbot_response += chunk['message']['content']
    st.write(chatbot_response, flush=True)

