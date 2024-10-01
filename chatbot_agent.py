import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# Load the API key from Streamlit secrets
openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=openai_api_key)

# Load the SentenceTransformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the FAISS index for similarity search
index = faiss.read_index("post_embeddings.index")

# Load the post titles or full texts (assuming you have them stored in a JSON file)
with open('wordpress_posts.json', 'r') as file:
    posts_data = json.load(file)

# Extract titles and links for the posts
post_titles = [post['title']['rendered'] for post in posts_data]
post_links = [post['link'] for post in posts_data]  # Assuming 'link' is the key for post URLs

# Streamlit app interface
st.title("Pet Blog Chatbot with Agents")

# User input
user_query = st.text_input("Ask your question:")

if user_query:
    # Generate embedding for the user's query
    query_embedding = model.encode([user_query])

    # Search the FAISS index for the top 5 most similar posts
    D, I = index.search(np.array(query_embedding), k=5)

    # Fetch the relevant post titles or full content
    relevant_posts = [(post_titles[i], post_links[i]) for i in I[0]]  # Now includes both title and link
    context = " ".join([post[0] for post in relevant_posts])  # Combine only titles for the context

    # Create a chat completion using the OpenAI client structure
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{context} {user_query}"}
        ],
        max_tokens=100  # Set the maximum tokens for the response
    )

    # Display the generated response from OpenAI
    st.write("Chatbot response:")
    st.write(chat_completion.choices[0].message.content.strip())

    # Optionally display the relevant posts used for the context as clickable links
    st.write("Relevant posts used:")
    for title, link in relevant_posts:
        st.markdown(f"- [{title}]({link})")  # Show title as a clickable link

    # Summarization feature
    st.write("Select the posts you want to summarize:")

    # Add checkboxes for each post
    selected_posts = []
    for title, link in relevant_posts:
        if st.checkbox(f"{title}", key=title):  # Each checkbox has a unique key
            selected_posts.append(title)

    # Summarize the selected posts
    if st.button("Summarize selected posts"):
        summaries = []
        for post in selected_posts:
            # Use chat completion for summarization as well
            summary_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": f"Summarize the following post: {post}"}
                ],
                max_tokens=150
            )
            summaries.append(f"Summary of {post}: {summary_response.choices[0].message.content.strip()}")

        st.write("Summaries:")
        for summary in summaries:
            st.write(summary)