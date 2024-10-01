import streamlit as st
import openai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from serpapi import GoogleSearch  # For fetching external information


# Load the API key from Streamlit secrets
openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]
openai.api_key = openai_api_key

# Initialize the OpenAI client with the API key
client = openai.OpenAI(
    api_key=openai_api_key
)

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
st.title("Pet Blog Chatbot")

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

    # Create a chat completion using the new OpenAI client structure
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

    # If posts are selected, show a summarize button
    if st.button("Summarize Selected Posts"):
        if selected_posts:
            # Example summarization logic using the new OpenAI ChatCompletion API
            summaries = []
            for title in selected_posts:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": f"Summarize this post: {title}"}
                    ],
                    max_tokens=100
                )
                summaries.append(f"Summary of {title}: {response.choices[0].message['content'].strip()}")

            st.write("Summaries:")
            for summary in summaries:
                st.write(summary)
        else:
            st.write("Please select at least one post to summarize.")

    # External Search Agent for fetching relevant information
    st.write("Fetching information from external sources...")

    # SerpAPI Google Search
    serpapi_api_key = st.secrets["general"]["SERPAPI_KEY"]  # Add SERPAPI API key to secrets
    params = {
        "q": user_query,  # Query based on the user's question
        "hl": "en",       # Language
        "gl": "us",       # Country
        "api_key": serpapi_api_key
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    # Display the search results
    if "organic_results" in results:
        st.write("External Information Fetched from Google:")
        for result in results["organic_results"][:5]:  # Show top 5 results
            st.markdown(f"- [{result['title']}]({result['link']})")
