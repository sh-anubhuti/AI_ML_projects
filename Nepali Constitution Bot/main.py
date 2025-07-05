import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
# --- Updated Imports for Google Gemini ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode

# --- Global variable for the compiled graph ---
# This ensures the graph is compiled only once.
RAG_GRAPH = None

def initialize_rag_graph():
    """
    Initializes all components and compiles the RAG graph using Google Gemini.
    This should be called once when the application starts.
    """
    global RAG_GRAPH
    if RAG_GRAPH is not None:
        print("Graph already initialized.")
        return

    # --- 1. Configuration and Setup ---
    print("Attempting to load environment variables...")
    load_dotenv(find_dotenv())
    # --- Switched to GOOGLE_API_KEY ---
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
    print("Google API Key loaded successfully.")

    # --- 2. Initialize Core Components (LLM and Embeddings) ---
    try:
        # --- Switched to Google's LLM and Embedding models ---
        # Using convert_system_message_to_human=True for compatibility with some graph structures
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key, convert_system_message_to_human=True)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        print("Google Gemini LLM and Embedding models initialized.")
    except Exception as e:
        raise RuntimeError(f"Error initializing Google Gemini models: {e}")

    # --- 3. Document Loading and Vector Store Setup ---
    pdf_path = "Data/Constitution-of-Nepal_2072.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The PDF file was not found at: {pdf_path}")

    print(f"Loading document from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    print(f"Document split into {len(all_splits)} chunks.")
    
    print("Creating FAISS vector store...")
    vector_store = FAISS.from_documents(all_splits, embeddings)
    print("Vector store created successfully.")

    # --- 4. Define the Retrieval Tool ---
    @tool
    def retrieve_from_document(query: str) -> str:
        """
        Retrieve relevant excerpts from the document based on the user's query.
        """
        print(f"--- TOOL: Retrieving documents for query: '{query}' ---")
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(query)
        context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)
        return context

    tools = [retrieve_from_document]
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    # --- 5. Define the Graph Nodes ---
    def agent_router(state: MessagesState) -> str:
        last_message = state['messages'][-1]
        return "call_tool" if last_message.tool_calls else "END"

    def generation_node(state: MessagesState) -> MessagesState:
        response = llm.invoke(state['messages'])
        return {"messages": [response]}

    def initial_responder_node(state: MessagesState) -> MessagesState:
        response = llm_with_tools.invoke(state['messages'])
        return {"messages": [response]}

    # --- 6. Construct the Graph ---
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("initial_responder", initial_responder_node)
    graph_builder.add_node("generate", generation_node)
    graph_builder.add_node("call_tool", tool_node)
    graph_builder.set_entry_point("initial_responder")
    graph_builder.add_conditional_edges(
        "initial_responder", agent_router, {"call_tool": "call_tool", "END": END}
    )
    graph_builder.add_edge("call_tool", "generate")
    graph_builder.add_edge("generate", END)

    RAG_GRAPH = graph_builder.compile()
    print("\nGraph compiled successfully!")

def get_response(user_input: str, conversation_history: list) -> str:
    """
    Takes user input and conversation history, invokes the graph, and returns the AI response.
    """
    if RAG_GRAPH is None:
        raise RuntimeError("Graph not initialized. Call initialize_rag_graph() first.")

    # Append the new user message to the history
    history_for_graph = conversation_history + [HumanMessage(content=user_input)]
    
    # Invoke the graph
    result = RAG_GRAPH.invoke({"messages": history_for_graph})
    
    # The result contains the full state. The final response is the last message.
    final_response = result['messages'][-1].content
    return final_response

# This part allows you to still run main.py directly for testing if you want
if __name__ == '__main__':
    print("Running main.py in interactive mode for testing.")
    initialize_rag_graph()
    
    # We will store the conversation history in this list.
    conversation_history: list[BaseMessage] = []
    
    print("\nStarting conversation. Type 'exit' to end.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break

        # Get response from our new function
        ai_response = get_response(user_input, conversation_history)
        
        # Update history for the next turn
        conversation_history.append(HumanMessage(content=user_input))
        conversation_history.append(AIMessage(content=ai_response))
        
        print(f"\nAI: {ai_response}")
