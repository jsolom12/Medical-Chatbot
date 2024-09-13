from langchain.document_loaders import PyPDFLoader, DirectoryLoader 
from langchain.text_splitter import  RecursiveCharacterTextSplitter  
from langchain.embeddings import HuggingFaceBgeEmbeddings 
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.llms import CTransformers

import os


def load_data(Data):
    loader = DirectoryLoader(Data, glob="*.pdf", loader_cls=PyPDFLoader)
    doc = loader.load()
    
    print(f"Number of documents loaded: {len(doc)}")
    
    # If documents are loaded, print a preview of the content
    if len(doc) > 0:
        for i, d in enumerate(doc):
            print(f"Document {i} content (first 500 chars): {d.page_content[:500]}...")
    else:
        print("No documents were loaded. Please check the file path and file format.")
    
    return doc



# Converting the corpus to chunks 
#Converts long documents into smaller chunks for better processing and storage in the vector database. For embedding generation and similarity-based searches.
def textSplit(data_extracted):
    if not data_extracted or len(data_extracted) == 0:
        raise ValueError("No documents found for splitting.")
    
    # Initialize the text splitter
    text_chunk_split = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    
    # Perform the text split
    chunk = text_chunk_split.split_documents(data_extracted)

    # Debugging: Output the number of chunks and their contents
    print(f"Number of chunks created: {len(chunk)}")
    if len(chunk) > 0:
        for i, ch in enumerate(chunk):
            print(f"Chunk {i} content: {ch.page_content[:200]}...")  # Print first 200 characters of each chunk
    
    return chunk


def getModelEmbedding():
    embdeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embdeddings


# os.environ["PINECONE_API_KEY"] = "45060129-0c79-4f9a-aa11-60095ca285b2"


# Function to initialize Pinecone
def initialize_pinecone(index_name):
    PINECONE_APIKEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = "us-east-1"
    
    # Initialize Pinecone using the instance-based method
    pc = Pinecone(api_key=PINECONE_APIKEY)

    # Check if the index exists, and create it if necessary
    if index_name not in pc.list_indexes().names():
        print(f"Creating index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region=PINECONE_ENV)
        )
    else:
        print(f"Using existing index: {index_name}")

    return pc.Index(index_name)



def store_embeddings(chunks_of_text, model_embedding, index_name):
    # Initialize Pinecone index
    pinecone_index = initialize_pinecone(index_name)

    # Generate embeddings for each chunk of text
    embeddings = model_embedding.embed_documents([chunk.page_content for chunk in chunks_of_text])

    # Prepare and upsert data into Pinecone
    for i, chunk in enumerate(chunks_of_text):
        # Each document is upserted with the embedding vector and metadata (the text)
        pinecone_index.upsert(
            vectors=[
                {
                    'id': str(i),  # Unique ID for each chunk
                    'values': embeddings[i],  # Embedding vector
                    'metadata': {'text': chunk.page_content}  # Store chunk content as metadata
                }
            ]
        )
    print("All embeddings have been successfully stored!")


def query_pinecone(index_name, user_query, model_embedding, llm_model):
    vectorstore = LangchainPinecone.from_existing_index(
        index_name=index_name,
        embedding=model_embedding 
    )
    
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="stuff",
        retriever=retriever
    )
    
    # Perform the query and get the result
    return qa_chain.run(user_query)

def chat_with_medbot(index_name, model_embedding, llm_model):
    while True:
        # Ask for user input
        query = input("Ask me a medical question (or type 'exit' to stop): ")

        # Break the loop if the user wants to exit
        if query.lower() == 'exit':
            print("Thank you for using the medical chatbot. Goodbye!")
            break

        # Run the query and get the response
        response = query_pinecone(index_name=index_name, user_query=query, model_embedding=model_embedding, llm_model=llm_model)

        # Print the response
        print("Response:", response)
