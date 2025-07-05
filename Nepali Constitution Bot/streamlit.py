import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# Import the functions from your main logic file
from main import initialize_rag_graph, get_response

# --- 1. App Configuration and Title ---
st.set_page_config(page_title="नेपाली संविधान बट", page_icon="🇳🇵", layout="wide")

# --- Updated Title to mention Gemini ---
st.title("नेपाली संविधान बट")
st.info("नेपालको संविधान (२०७२) बारे कुनै पनि प्रश्न सोध्नुहोस्। बटले सान्दर्भिक लेखहरू फेला पार्नेछ र जवाफ दिनेछ।")

# --- 2. Sidebar for Chat History ---
with st.sidebar:
    st.title("Chat History")
    
    # This will be populated with the user's questions
    history_container = st.container()


# --- 3. Initialization ---
# Use session state to ensure initialization happens only once per session
if "initialized" not in st.session_state:
    with st.spinner("We are getting ready for you!!"):
        try:
            # Initialize the RAG graph from main.py
            initialize_rag_graph()
            st.session_state.initialized = True
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            # --- Updated Error Message ---
            st.error(f"Initialization Failed: {e}. Please ensure your GOOGLE_API_KEY is set correctly in the .env file.")
            st.stop() # Stop the app if initialization fails

# --- 4. Chat Interface ---
# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages in the main chat area
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Update the history in the sidebar
with history_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.info(f"You: {message['content'][:50]}...") # Display a snippet of user's query


# Handle user input
if prompt := st.chat_input("Ask us your questions..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Prepare the history in the format expected by the get_response function
            history_for_graph = []
            # We don't include the current prompt in the history list passed to the function
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    history_for_graph.append(HumanMessage(content=msg["content"]))
                else:
                    history_for_graph.append(AIMessage(content=msg["content"]))

            # Call the get_response function from main.py
            response = get_response(prompt, history_for_graph)
            st.markdown(response)

    # Add AI response to the session state history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun the script to update the sidebar immediately
    st.rerun()
