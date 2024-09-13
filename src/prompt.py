# Import necessary functions from helper.py and any other required libraries
from src.helper import query_pinecone

# Function to interact with the medical chatbot via command line (CLI)
def chat_with_medbot(index_name, model_embedding, llm_model):
    """
    Function to handle user interaction with the medical chatbot via CLI.
    
    Parameters:
    - index_name: The name of the Pinecone index.
    - model_embedding: The embedding model to convert user queries into embeddings.
    - llm_model: The language model used to generate responses.
    """
    while True:
        # Prompt user for a medical question
        query = input("Ask me a medical question (or type 'exit' to stop): ")

        # If the user wants to exit, break the loop
        if query.lower() == 'exit':
            print("Thank you for using the medical chatbot. Goodbye!")
            break

        # Get the response from Pinecone using the user's query
        response = query_pinecone(
            index_name=index_name,
            user_query=query,
            model_embedding=model_embedding,
            llm_model=llm_model
        )

        # Print the chatbot's response
        print("Response:", response)


# Function to get a response from the chatbot (used by web app)
def get_response(index_name, user_query, model_embedding, llm_model):
    """
    Function to get a response from the chatbot.
    
    Parameters:
    - index_name: The name of the Pinecone index.
    - user_query: The question asked by the user.
    - model_embedding: The embedding model to convert the user's query.
    - llm_model: The language model used to generate responses.
    
    Returns:
    - response: The chatbot's response to the query.
    """
    response = query_pinecone(
        index_name=index_name,
        user_query=user_query,
        model_embedding=model_embedding,
        llm_model=llm_model
    )
    return response
