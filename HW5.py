import streamlit as st
import openai
import json
import numpy as np

# Load the OpenAI API key from Streamlit's secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load data (e.g., courses.json or clubs.json)
data_path = "data/courses.json"  # Change this path if you're working with clubs.json
with open(data_path, "r") as file:
    course_data = json.load(file)

# Function to embed the course/club descriptions using OpenAI's API
def embed_course_data(course_data):
    embeddings = {}
    for course in course_data:
        description = course["description"]
        embedding = get_embedding(description)
        embeddings[course["name"]] = embedding
    return embeddings

# Function to get embedding using OpenAI API
def get_embedding(text):
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    return embedding

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Function to handle vector search
def vector_search(query, course_data, embeddings):
    query_embedding = get_embedding(query)
    results = []
    
    for course_name, course_embedding in embeddings.items():
        similarity = cosine_similarity(query_embedding, course_embedding)
        results.append((course_name, similarity))
    
    # Results are sorted by similarity
    results = sorted(results, key=lambda x: x[1], reverse=True)
    top_results = [course for course, _ in results[:3]]  # Get top 3 relevant courses
    return top_results

# Short-term memory chatbot
def chatbot_interaction(query, course_data, embeddings):
    relevant_courses = vector_search(query, course_data, embeddings)
    
    # Retrieve information about the top 3 courses
    relevant_course_info = ""
    for course in relevant_courses:
        course_info = next((item for item in course_data if item["name"] == course), {})
        relevant_course_info += f"Course Name: {course_info['name']}\nDescription: {course_info['description']}\n\n"
    
    # Use the relevant course info as context in the LLM response
    system_prompt = f"You are a helpful assistant. You have access to the following course information: {relevant_course_info}"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error with OpenAI completion: {e}")
        return "An error occurred while processing your request."

# Streamlit App
st.title("Short-Term Memory Chatbot with Course Info")

st.write("Ask a question and I will retrieve relevant course information:")

# Load embeddings (or generate if not already done)
if "embeddings" not in st.session_state:
    st.session_state.embeddings = embed_course_data(course_data)

# User query input
query = st.text_input("Enter your question:")

if query:
    response = chatbot_interaction(query, course_data, st.session_state.embeddings)
    st.write("Response:", response)

# File Upload Feature
st.write("You can also upload additional course data to update the system:")
uploaded_file = st.file_uploader("Choose a file", type=["json"])

if uploaded_file is not None:
    uploaded_data = json.load(uploaded_file)
    st.write("New data uploaded successfully!")
    
    # Update the course data with the uploaded file and regenerate embeddings
    course_data.extend(uploaded_data)
    st.session_state.embeddings = embed_course_data(course_data)
