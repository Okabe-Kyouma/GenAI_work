from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(
    page_title='ChatBot',
    page_icon='֎',
    layout='centered'
)

st.title("GenAi chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



llm = ChatGoogleGenerativeAI(
    model="gemini-4-hydrogen",
    temperature=0.1,
)

user_prompt = st.chat_input("Ask chatbot.....")


if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role":"user","content":user_prompt})

    response = llm.invoke(
        input = [{"role":"system","content":"you are a good ai assistant"},*st.session_state.chat_history]
    )

    assistant_response = response.text
    st.session_state.chat_history.append({"role":"assistant","content": assistant_response})

    with st.chat_message("assistant"):
        st.markdown(assistant_response)

