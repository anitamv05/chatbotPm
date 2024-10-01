import streamlit as st
import openai
from serpapi import GoogleSearch

# Function for performing SerpAPI search
def serpapi_search(query):
    params = {
        "q": query,
        "location": "United States",
        "api_key": st.secrets["SERPAPI_KEY"]  # Place your SerpAPI key in secrets.toml
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    return results

# Load OpenAI key from secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit UI for taking user input
st.title("Pet Blog Chatbot with External Search Agents")
user_query = st.text_input("Ask your question:")

if user_query:
    # Use OpenAI API to generate response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_query}],
        max_tokens=100
    )

    # Display the generated response from OpenAI
    st.write("Chatbot response:")
    st.write(response.choices[0].message.content.strip())

    # External Search using SerpAPI
    st.write("Fetching relevant search results from the web...")
    search_results = serpapi_search(user_query)

    # Display top 3 search results
    st.write("Top 3 Search Results:")
    for result in search_results.get("organic_results", [])[:3]:
        st.write(f"[{result['title']}]({result['link']})")

    # Additional features (optional):
    # - Summarize search results using OpenAI
    # - Provide more detailed information about each result (e.g., description, author)
    # - Allow users to refine their search queryc
    