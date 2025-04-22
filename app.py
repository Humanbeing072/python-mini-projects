import streamlit as st
from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

st.set_page_config(page_title='Conversational Q&A Chatbot')
st.header("Hey,let's Chat!")

from dotenv import load_dotenv
load_dotenv()
import os
print(os.environ.get('GOOGLE_API_KEY'))
chat = ChatGoogleGenerativeAI(
    google_api_key=os.environ['GOOGLE_API_KEY'], 
    model="gemini-1.5-flash", 
    temperature=0.5
)

if 'flowmessage' not in st.session_state:
    st.session_state['flowmessage'] = [
        SystemMessage(content="You are narcissistic AI assistant"),
        HumanMessage(content="Your question here")
    ]

def get_chatmodel_response(question):
    st.session_state['flowmessage'].append(HumanMessage(content=question))  # Corrected key
    answer = chat(st.session_state['flowmessage'])  # Corrected key
    st.session_state['flowmessage'].append(AIMessage(content=answer.content))  # Corrected key
    print(st.session_state['flowmessage'])
    return answer.content

input = st.text_input("Input", key="input")
if input.strip():
    response = get_chatmodel_response(input)
else:
    response = "Please enter a valid question."

submit = st.button("Ask the question")
if submit:
    st.subheader("The Response is")
    st.write(response)