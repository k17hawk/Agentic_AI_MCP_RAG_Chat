from dotenv import load_dotenv
import os
import requests
import json
import streamlit as st

load_dotenv()

# Streamlit page setup
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="centered",
)
st.title("💬 Generative AI Chatbot")

# Chat history in streamlit session
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Show chat history
for message in st.session_state.messages[1:]:  # Skip system message
    with st.chat_message(message['role']):
        st.markdown(message['content'])

def get_ollama_response(prompt, history):
    """Get response from Ollama API"""
    url = "http://localhost:11434/api/chat"
    
    # Prepare messages
    messages = history + [{"role": "user", "content": prompt}]
    
    payload = {
        "model": "llama3:latest",  # Change to your preferred model
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure Ollama is running on localhost:11434"
    except Exception as e:
        return f"Error: {str(e)}"

# Input box
user_prompt = st.chat_input("Ask Chatbot...")
if user_prompt:
    # Add user message to history
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({'role': "user", "content": user_prompt})
    
    # Get response from Ollama
    assistant_response = get_ollama_response(user_prompt, st.session_state.messages)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)