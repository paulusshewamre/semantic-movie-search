import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


#load model
@st.cache_resource
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

model = load_model()

#load data
@st.cache_data
def load_data():
    # Replace the path with your downloaded Kaggle CSV file
    # Example: "datasets/IMDB Dataset.csv"
    df = pd.read_csv("IMDB Dataset.csv")  
    # Keep only the first 500 rows for demo (optional)
    df = df.head(500)
    df['review'] = df['review'].str.replace('<br />', ' ', regex=False)
    # Use the 'review' column as documents
    documents = df['review'].tolist()
    return documents

documents = load_data()

# Precompute document embeddings
@st.cache_resource
def compute_embeddings(documents):
    return model.encode(documents, show_progress_bar=True)

doc_embeddings = compute_embeddings(documents)

# Streamlit UI
st.set_page_config(page_title="Semantic Search Demo", layout="wide")
st.title("🔍 Semantic Search Engine")
st.write("Type a question or keyword, and the system will find the most semantically relevant documents from IMDB reviews.")


# User input
query = st.text_input("Enter your query:", placeholder="e.g., a movie with great acting")

if query:
    # Encode query
    query_embedding = model.encode([query])
    
    # Compute similarities
    similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()
    top_indices = np.argsort(similarities)[::-1]

    # Display top results
    st.subheader("Top Relevant Reviews:")
    for idx in top_indices[:5]:  # Show top 5
        st.markdown(f"""
        **Review:** {documents[idx]}  
        **Similarity:** {similarities[idx]:.3f}
        ---
        """)

st.sidebar.header("About")
st.sidebar.markdown("""
This is a **semantic search app** built with:
- [Sentence Transformers](https://www.sbert.net/)
- [Streamlit](https://streamlit.io)
- [Scikit-learn](https://scikit-learn.org)
- Kaggle dataset: [IMDB Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
""")
