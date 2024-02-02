import os
import sys
import requests
import streamlit as st
from github import Github
from bs4 import BeautifulSoup
from datetime import datetime
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter

if "OPENAI_API_KEY" not in st.secrets:
    st.error("Please set the OPENAI_API_KEY secret on the Streamlit dashboard.")
    sys.exit(1)

openai_api_key = st.secrets["OPENAI_API_KEY"]

g = Github(st.secrets["GITHUB_TOKEN"])
repo = g.get_repo("scooter7/urlchatbot")

def setup_content_dir(folder):
    content_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "content", folder)
    os.makedirs(content_dir, exist_ok=True)
    st.session_state.content_dir = content_dir
    st.session_state.content_path = "content/" + folder

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.directory_path = directory_path
    index.save_to_disk('index.json')

def append_to_chat_history(question, answer):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append({'user': question, 'bot': answer})

def extract_text_from(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(line for line in lines if line)

@st.cache
def create_index_from_urls(urls):
    pages = []
    for url in urls:
        pages.append({'text': extract_text_from(url), 'source': url})

    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    docs, metadatas = [], []
    for page in pages:
        splits = text_splitter.split_text(page['text'])
        docs.extend(splits)
        metadatas.extend([{"source": page['source']}] * len(splits))

    data_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs/data.txt")
    with open(data_file_path, 'w') as f:
        for page in pages:
            f.write(page['text'] + '\n')

@st.cache(ttl=120)
def standard_responses():
    std_responses = {}
    with open('info/standard.txt', 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            line = line.strip()
            if line != '':
                key, value = line.split('::')
                std_responses[key] = value.strip()

    return std_responses

def standard_response(k):
    std_responses = standard_responses()
    if k in std_responses:
        return std_responses[k]
    else:
        return False

@st.cache(ttl=120)
def counselor_responses():
    clr_responses = []
    with open('info/counselor_keywords.txt', 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            line = line.strip()
            if line != '':
                clr_responses.append(line.lower())

    return clr_responses

def is_counselor_response(k):
    clr_responses = counselor_responses()
    for resp in clr_responses:
        if resp in k:
            return True
    return False

def chatbot(input_text, full_name = '', email = ''):
    construct_index('docs')
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    prompt = input_text

    std_r = standard_response(prompt.lower())
    is_clr_r = is_counselor_response(prompt.lower())

    if std_r:
        response = std_r
    elif is_clr_r:
        response = 'Admissions counselor will follow up soon. Thanks!'
    else:
        response = index.query(prompt, response_mode="compact").response

    append_to_chat_history(prompt, response)

    if 'filename' not in st.session_state:
        st.session_state.filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.txt")

    filename = st.session_state.filename
    file_path = os.path.join(st.session_state.content_dir, filename)
    with open(file_path, 'a') as f:
        user_input_prefix = full_name if full_name != '' else 'Input'
        if email != '':
            user_input_prefix += ' (' + email + ')'
        f.write(f"{user_input_prefix}: {input_text}\n")
        f.write(f"Chatbot response: {response}\n")

    with open(file_path, 'rb') as f:
        contents = f.read()
        if 'github_file_created' not in st.session_state:
            st.session_state.github_file_created = False

        if st.session_state.github_file_created:
            repo.update_file(f"{st.session_state.content_path}/{filename}", f"Update chat file {filename}", contents, repo.get_contents(f"{st.session_state.content_path}/{filename}").sha)
        else:
            repo.create_file(f"{st.session_state.content_path}/{filename}", f"Add chat file {filename}", contents)
            st.session_state.github_file_created = True

def hide_branding():
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)
