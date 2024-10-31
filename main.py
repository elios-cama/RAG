import streamlit as st
import os
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_huggingface import HuggingFaceEmbeddings

# Custom CSS for dark theme with improved chat UI
st.markdown("""
<style>
    body {
        color: #FFFFFF;
        background-color: #000000;
    }
    .stTextInput > div > div > input {
        color: #FFFFFF;
        background-color: #333333;
    }
    .stButton > button {
        color: #FFFFFF;
        background-color: #0E86D4;
        border: none;
        border-radius: 20px;
    }
    .stButton > button:hover {
        background-color: #055C9D;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.bot {
        background-color: #475063;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .avatar img {
        max-width: 78px;
        max-height: 78px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .message {
        width: 80%;
        padding: 0 1.5rem;
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)

# Function to prepare and split documents
def prepare_and_split_docs(pdf_directory):
    loader = DirectoryLoader(pdf_directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=256,
        disallowed_special=(),
        separators=["\n\n", "\n", " "]
    )
    return splitter.split_documents(documents)

# Function to ingest documents into the vector database
def ingest_into_vectordb(split_docs):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    db = FAISS.from_documents(split_docs, embeddings)
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)
    return db

# Function to get the conversation chain
def get_conversation_chain(retriever):
    llm = Ollama(model="llama3.2")
    contextualize_q_system_prompt = (
        "Given the chat history and the latest user question, "
        "provide a response that directly addresses the user's query based on the provided documents. "
        "Do not rephrase the question or ask follow-up questions."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are Tadz, the AI assistant for our company event. Your role is to provide helpful information "
        "about the event schedule, speakers, venues, and answer any questions attendees might have. "
        "Be concise, friendly, and always aim to provide the most relevant information from the event documents. "
        "If you're unsure about something, politely suggest the attendee speak with one of our event staff for more details. "
        "Limit your responses to 2-3 sentences and about 50 words for clarity. "
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

# Initialize session state
if 'conversational_chain' not in st.session_state:
    # Process documents and create conversation chain on first run
    pdf_directory = "assets"  # Update this path to your project's PDF directory
    if os.path.exists(pdf_directory):
        split_docs = prepare_and_split_docs(pdf_directory)
        vector_db = ingest_into_vectordb(split_docs)
        retriever = vector_db.as_retriever()
        st.session_state.conversational_chain = get_conversation_chain(retriever)
    else:
        st.error(f"PDF directory not found: {pdf_directory}")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("Chat with Tadz - Your Event Assistant 🤖")

# Chat interface
st.markdown("### Ask Tadz about the event!")
user_input = st.text_input("Your question:")

if st.button("Ask"):
    if user_input and 'conversational_chain' in st.session_state:
        session_id = "event_chat"
        conversational_chain = st.session_state.conversational_chain
        response = conversational_chain.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
        bot_response = response['answer']
        st.session_state.chat_history.append({"user": user_input, "bot": bot_response})

# Display chat history
for message in st.session_state.chat_history:
    if message["user"]:
        st.markdown(f'<div class="chat-message user"><div class="avatar"><img src="https://docs.ta-da.io/~gitbook/image?url=https%3A%2F%2F1998936910-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252F36WmpDYmmWMKo40O88Gq%252Fuploads%252FB8aVZDjXTtjnbSVjKxTi%252Ftest1.jpg%3Falt%3Dmedia%26token%3D18836d9c-e360-47d1-896a-dffd84792dcb&width=300&dpr=2&quality=100&sign=51fdc13d&sv=1"/></div><div class="message">{message["user"]}</div></div>', unsafe_allow_html=True)
    if message["bot"]:
        st.markdown(f'<div class="chat-message bot"><div class="avatar"><img src="https://pbs.twimg.com/profile_images/1812154656342065152/8EI4kI0A_400x400.jpg"/></div><div class="message">{message["bot"]}</div></div>', unsafe_allow_html=True)

st.markdown("### Enjoy the event! 🎉")