import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


load_dotenv()

st.set_page_config(page_title="Multi-select Gen AI Chatbot", page_icon=":robot_face:",layout="centered")
st.title("Multi-select Gen AI Chatbot")



if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


llms = {
    "Gemini" : ChatGoogleGenerativeAI(
    model="gemini-4-hydrogen",
    temperature=0.1,
),

    "GPT-6 Luna" : ChatOpenAI(
        model_name="gpt-6-luna",
    )
}


user_prompt = st.chat_input("Ask me anything!")

selected_llms = st.multiselect(
    "Select LLM:",
    options=list(llms.keys()),
    default=["Gemini"]
)

if user_prompt:

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_prompt
    })

    with st.chat_message("user"):
        st.markdown(user_prompt)

    if not selected_llms:
        st.warning("Please select at least one model.")

    else:
        for model_name in selected_llms:

            with st.chat_message("assistant"):
                with st.spinner(f"{model_name} is thinking..."):

                    response = llms[model_name].invoke(
                        [
                            {
                                "role": "system",
                                "content": "You are a helpful AI assistant."
                            },
                            *st.session_state.chat_history
                        ]
                    )

                    assistant_response = response.content

                    st.markdown(f"**{model_name}**")
                    st.markdown(assistant_response)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"[{model_name}] {assistant_response}"
                    })