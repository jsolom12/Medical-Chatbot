import os
from dotenv import load_dotenv
from src.helper import load_data, textSplit, getModelEmbedding, initialize_pinecone, store_embeddings

# Load environment variables (Pinecone API Key and Environment) from .env file
load_dotenv()

# Retrieve the Pinecone API Key and Environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Debugging: Print the API key and environment to verify they are being loaded
print(f"Pinecone API Key: {PINECONE_API_KEY}")
print(f"Pinecone Environment: {PINECONE_ENV}")

# Ensure that API Key and Environment are set
if not PINECONE_API_KEY or not PINECONE_ENV:
    raise ValueError("Pinecone API key or environment is not set properly.")

# Define Pinecone Index Name
index_name = "medchatbot"

# Load Data from PDF files
data_directory = "Data/"  # Change this to the actual path of your PDF files
print(f"Loading data from directory: {data_directory}")
documents = load_data(data_directory)

# Split the documents into smaller chunks for embedding
print(f"Splitting documents into chunks...")
chunks_of_text = textSplit(documents)

# Initialize Pinecone Index
print(f"Initializing Pinecone index: {index_name}")
pinecone_index = initialize_pinecone(index_name)

# Get Hugging Face Embeddings Model
print(f"Loading embedding model...")
model_embedding = getModelEmbedding()

# Step 5: Store the embeddings into Pinecone
print(f"Storing embeddings into Pinecone index: {index_name}")
store_embeddings(chunks_of_text, model_embedding, index_name)

print("Embeddings successfully stored in the Pinecone index.")

