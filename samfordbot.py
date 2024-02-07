from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.chains import ConversationalRetrievalChain
from langchain_community.prompts import PromptTemplate
import streamlit as st
from streamlit_chat import message
from github import Github
import os

def load_data(urls):
    loaders = UnstructuredURLLoader(urls=urls)
    data = loaders.load()
    return data

def initialize_chain(docs, temperature, k):
    embeddings = HuggingFaceEmbeddings()
    vectorStore = FAISS.from_documents(docs, embeddings)
    retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": temperature, "truncation": True})
    QUESTION_PROMPT = PromptTemplate.from_template("""
        Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
        This is a conversation with a human. Answer the questions you get based on the knowledge you have.
        If you don't know the answer, just say that you don't, don't try to make up an answer.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:
        """)
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever,
                                               condense_question_prompt=QUESTION_PROMPT,
                                               return_source_documents=False, verbose=False)
    return qa

def conversational_chat(qa, query):
    result = qa({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def save_chat_history_to_github():
    g = Github(st.secrets["GITHUB_TOKEN"])
    repo = g.get_repo("scooter7/urlchatbot")
    chat_history_path = "content/chat_history.txt"  
    chat_history_content = "\n".join([f"{pair[0]}: {pair[1]}" for pair in st.session_state['history']])
    try:
        contents = repo.get_contents(chat_history_path)
        repo.update_file(contents.path, "Updating chat history", chat_history_content, contents.sha)
        st.success("Chat history updated successfully in GitHub!")
    except Exception as e:
        repo.create_file(chat_history_path, "Creating chat history", chat_history_content)
        st.success("Chat history created successfully in GitHub!")

def main():
    st.markdown("<h1 style='text-align: center;'>J.A.R.V.I.S</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Your Virtual Web Companion!</h3>", unsafe_allow_html=True)
    st.write("Meet J.A.R.V.I.S: Your interactive browsing companion. Seamlessly engage with websites using AI. Explore, question, and navigate—transforming your browsing experience into conversations and discovery!")
    predefined_urls = ["https://example.com/document1", "https://example.com/document2"]
    data = load_data(predefined_urls)
    docs = data  # Assuming 'docs' needs to be directly used without splitting
    temperature = 0.5
    k_value = 3
    qa = initialize_chain(docs, temperature, k_value)
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'user_info' not in st.session_state:
        st.session_state['user_info'] = {'full_name': '', 'email': ''}
    with st.form(key='user_info_form'):
        st.session_state['user_info']['full_name'] = st.text_input("Full Name:", key='full_name')
        st.session_state['user_info']['email'] = st.text_input("Email Address:", key='email')
        submit_user_info = st.form_submit_button(label='Submit Information')
    if not submit_user_info or not st.session_state['user_info']['full_name'] or not st.session_state['user_info']['email']:
        st.stop()
    response_container = st.container()
    container = st.container()
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Enter question", key='input')
            submit_button = st.form_submit_button(label='Send')
        if submit_button and user_input:
            output = conversational_chat(qa, user_input)
            save_chat_history_to_github()
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i))

if __name__ == "__main__":
    main()
