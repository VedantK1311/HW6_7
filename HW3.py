import logging

# Suppress sagemaker info messages globally
logging.getLogger("sagemaker").setLevel(logging.ERROR)

# Now import the other libraries
import streamlit as st
from openai import OpenAI
import tiktoken
import cohere
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup


def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

def count_tokens(messages, model):
    if model.startswith("gpt"):
        encoding = tiktoken.encoding_for_model(model)
        return sum(len(encoding.encode(message["content"])) for message in messages)
    elif model.startswith("gemini"):
        # Approximate token count for Gemini (4 characters ~= 1 token)
        return sum(len(message["content"]) // 4 for message in messages)
    else:
        # For other models, use a simple character count as approximation
        return sum(len(message["content"]) for message in messages)

def update_conversation_buffer(max_tokens, model):
    full_history = st.session_state.messages
    current_tokens = count_tokens(full_history, model)
    while current_tokens > max_tokens and len(full_history) > 1:
        full_history.pop(0)
        current_tokens = count_tokens(full_history, model)
    st.session_state.messages = full_history

def stream_response(response):
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

def generate_response(client, model, messages, prompt):
    if model.startswith("gpt"):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        return "stream", response
    elif model.startswith("gemini"):
        response = client.generate_content(prompt)
        return "text", response.text
    elif model.startswith("command"):
        response = client.generate(prompt=prompt)
        return "text", response.generations[0].text

def hw3():
    st.title("Streaming Chatbot by Deep")

    st.sidebar.title("Chatbot Options")
    url1 = st.sidebar.text_input("Enter 1st URL")
    url2 = st.sidebar.text_input("Enter 2nd URL")

    if url1:
        st.sidebar.write(f"You entered: {url1}")
        st.session_state.url1_content = read_url_content(url1)
    if url2:
        st.sidebar.write(f"You entered: {url2}")
        st.session_state.url2_content = read_url_content(url2)

    conversation_memory = st.sidebar.selectbox(
        "Choose the type of conversation buffer:",
        ["Buffer of 5 questions", "Conversation Summary", "Buffer of 5000 tokens"]
    )

    model_selected = st.sidebar.selectbox(
        "Choose the LLM Model:",
        ["GPT", "Gemini", "Cohere"]
    )

    advanced_version = st.sidebar.checkbox("Use the advanced version")

    models = {
        "GPT": "gpt-4o-mini",
        "GPT-Advanced": "gpt-4o",
        "Gemini": "gemini-1.5-flash",
        "Gemini-Advanced": "gemini-1.5-pro",
        "Cohere": "command-r",
        "Cohere-Advanced": "command-r-plus"
    }

    model_to_use = f"{model_selected}-Advanced" if advanced_version else model_selected

    if 'client' not in st.session_state:
        st.session_state.client = {}
        gpt_key = st.secrets["gpt_key"]
        gemini_key = st.secrets["gemini_key"]
        cohere_key = st.secrets["cohere_key"]
        
        st.session_state.client["GPT"] = OpenAI(api_key=gpt_key)
        genai.configure(api_key=gemini_key)
        st.session_state.client["Gemini"] = genai.GenerativeModel(models[model_to_use])
        st.session_state.client["Cohere"] = cohere.Client(api_key=cohere_key)

    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you today?"}]
    
    if 'full_messages' not in st.session_state:
        st.session_state.full_messages = [{"role": "assistant", "content": "Hi! How can I help you today?"}]
    
    st.session_state.awaiting_info_reply = st.session_state.get("awaiting_info_reply", False)
    st.session_state.last_question = st.session_state.get("last_question", None)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to ask?"):
        if st.session_state.awaiting_info_reply:
            if prompt.lower() == 'yes':
                last_question = st.session_state.last_question
                if last_question:
                    question = "Please provide more indepth details."
                    additional_info_prompt = f"{question} Refer to {last_question}. Explain in extremely simple terms."
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.session_state.full_messages.append({"role": "user", "content": additional_info_prompt})
                    with st.chat_message("user"):
                        st.markdown(question)
                    
                    client = st.session_state.client[model_selected]
                    model = models[model_to_use]

                    if model_selected == "Gemini":
                        client = genai.GenerativeModel(model)

                    response_type, response = generate_response(
                        client,
                        model,
                        st.session_state.full_messages,
                        additional_info_prompt
                    )

                    with st.chat_message("assistant"):
                        if response_type == "stream":
                            response_placeholder = st.empty()
                            full_response = ""
                            for chunk in stream_response(response):
                                full_response += chunk
                                response_placeholder.markdown(full_response + "▌")
                            response_placeholder.markdown(full_response)
                        else:
                            st.markdown(response)
                            full_response = response
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.session_state.full_messages.append({"role": "assistant", "content": full_response})
            elif prompt.lower() == 'no':
                st.session_state.awaiting_info_reply = False
                new_question_prompt = "What question can I help you with next?"
                st.session_state.messages.append({"role": "assistant", "content": new_question_prompt})
                st.session_state.full_messages.append({"role": "assistant", "content": new_question_prompt})
                with st.chat_message("assistant"):
                    st.markdown(new_question_prompt)
        else:
            url_context = ""
            if 'url1_content' in st.session_state:
                url_context += f"Context from URL 1: {st.session_state.url1_content}\n"
            if 'url2_content' in st.session_state:
                url_context += f"Context from URL 2: {st.session_state.url2_content}\n"
            
            full_prompt = f"{url_context}\n\nUser question: {prompt}\nPlease answer in separate paragraphs based on the provided context and use simple terms."
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.full_messages.append({"role": "user", "content": full_prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            max_tokens = 5000 if conversation_memory == "Buffer of 5000 tokens" else 512
            update_conversation_buffer(max_tokens, models[model_to_use])

            client = st.session_state.client[model_selected]
            model = models[model_to_use]

            if model_selected == "Gemini":
                client = genai.GenerativeModel(model)

            response_type, response = generate_response(
                client,
                model,
                st.session_state.full_messages,
                full_prompt
            )

            with st.chat_message("assistant"):
                if response_type == "stream":
                    response_placeholder = st.empty()
                    full_response = ""
                    for chunk in stream_response(response):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                else:
                    st.markdown(response)
                    full_response = response

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.full_messages.append({"role": "assistant", "content": full_response})

            st.session_state.last_question = full_prompt
            follow_up = "DO YOU WANT MORE INFO?"
            st.session_state.messages.append({"role": "assistant", "content": follow_up})
            st.session_state.full_messages.append({"role": "assistant", "content": follow_up})
            with st.chat_message("assistant"):
                st.markdown(follow_up)
            st.session_state.awaiting_info_reply = True

# if __name__ == "__main__":
#     hw3()
