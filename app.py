from flask import Flask, render_template, jsonify, request
from src.helper import getModelEmbedding, initialize_pinecone, textSplit, store_embeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Retrieve the Pinecone API Key and Environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Ensure that API Key and Environment are set
if not PINECONE_API_KEY or not PINECONE_ENV:
    raise ValueError("Pinecone API key or environment is not set properly.")

# Initialize Pinecone and LLM model outside the routes
index_name = "medchatbot"

# Initialize Pinecone
print(f"Initializing Pinecone index: {index_name}")
initialize_pinecone(index_name)

# Load the embedding model (e.g., Hugging Face model)
print("Loading the embedding model...")
model_embedding = getModelEmbedding()

# Initialize the LLM model
llm_model = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",  
    model_type="llama",                    
    max_new_tokens=512,                    
    temperature=0.9                         
)

# Initialize vectorstore for RetrievalQA
vectorstore = LangchainPinecone.from_existing_index(
    index_name=index_name,
    embedding=model_embedding
)

# Build the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_model,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Define the home route
@app.route('/')
def home():
    return render_template('chat.html')  # 'chat.html' should be located in the 'templates' folder

# API route to get a response from the chatbot
@app.route('/ask', methods=['POST'])
def ask_question():
    user_query = request.json.get('question', '')
    
    if user_query:
        # Get the chatbot's response using RetrievalQA
        response = qa_chain.run(user_query)
        return jsonify({'response': response})
    else:
        return jsonify({'error': 'No question provided'}), 400

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
