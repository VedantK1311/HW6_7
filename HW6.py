import pandas as pd
import datetime
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# Set up OpenAI API key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load and process data
@st.cache_data
def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)  # Ensure dates are UTC
    df['days_since_2000'] = (df['Date'] - pd.Timestamp('2000-01-01', tz='UTC')).dt.days
    return df

# Rank news items
def rank_news_items(df, keywords):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['Document'])
    keyword_vector = vectorizer.transform([' '.join(keywords)])
    similarities = cosine_similarity(keyword_vector, tfidf_matrix).flatten()
    df['relevance_score'] = similarities
    return df.sort_values('relevance_score', ascending=False)

# Search news
def search_news(df, query):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['Document'])
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    df['search_score'] = similarities
    return df.sort_values('search_score', ascending=False)

# Provide context using OpenAI
def provide_context(news_item):
    prompt = f"""
    As a legal expert, provide a brief analysis of the following news item:
    
    {news_item}
    
    Consider the potential legal implications, risks, and opportunities for a global law firm.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a legal expert providing brief analyses of news items."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# Main bot function
def news_bot(query, news_data):
    legal_keywords = ['legal', 'lawsuit', 'regulation', 'compliance', 'litigation', 'court', 'judge', 'attorney', 'law']
    if query.lower() == 'find the most interesting news':
        ranked_news = rank_news_items(news_data, legal_keywords)
        return ranked_news.head(5), "Most Interesting News"
    elif query.lower().startswith('find news about'):
        topic = query[16:].strip()
        search_results = search_news(news_data, topic)
        return search_results.head(5), f"News About {topic}"
    else:
        return None, "I'm sorry, I don't understand that query. Please try 'find the most interesting news' or 'find news about [topic]'."

# Streamlit app
def main():
    st.title("Legal News Bot")

    news_data = load_and_process_data('Example_news_info_for_testing.csv')

    query = st.text_input("Enter your query:")
    if st.button("Submit"):
        results, title = news_bot(query, news_data)
        
        if results is not None:
            st.header(title)
            for _, row in results.iterrows():
                st.subheader(f"{row['company_name']} - {row['Date'].strftime('%Y-%m-%d')}")
                st.write(row['Document'])
                with st.expander("Legal Context"):
                    st.write(provide_context(row['Document']))
        else:
            st.write(title)

if __name__ == "__main__":
    main()
