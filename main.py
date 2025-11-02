import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
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
    df = pd.read_csv("./data/IMDB Dataset.csv")
    df = df.head(500)
    df['review'] = df['review'].str.replace('<br />', ' ', regex=False)
    documents = df['review'].tolist()
    return documents

documents = load_data()


# Connect to Qdrant 
@st.cache_resource
def get_qdrant_client():
    client = QdrantClient(host="localhost", port=6333)
    return client

client = get_qdrant_client()

collection_name = "imdb_reviews"


# Create Collection if Not Exists
if collection_name not in [col.name for col in client.get_collections().collections]:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )


# Upload embeddings to Qdrant (only once)
@st.cache_resource
def upload_embeddings():
    doc_embeddings = model.encode(documents, show_progress_bar=True)

    # Prepare points for upload
    points = [
        PointStruct(id=i, vector=doc_embeddings[i].tolist(), payload={"review": documents[i]})
        for i in range(len(documents))
    ]

    # Upsert to Qdrant
    client.upsert(collection_name=collection_name, points=points)
    return True

upload_embeddings()


# Streamlit App
st.set_page_config(page_title="Semantic Search with Qdrant", layout="wide")
st.title("🔍 Semantic Search Engine (with Qdrant)")
st.write("Type a query below to find the most semantically relevant IMDB reviews.")


# querying qdrant
query = st.text_input("Enter your query:", placeholder="e.g., a movie with great acting")

if query:
    query_vector = model.encode([query])[0].tolist()

    # Search in Qdrant
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5,
    )

    # Display top results
    st.subheader("Top Relevant Reviews:")
    for hit in results:
        st.markdown(f"""
        **Review:** {hit.payload['review']}  
        **Score:** {hit.score:.3f}
        ---
        """)

st.sidebar.header("About")
st.sidebar.markdown("""
This is a **semantic search app** built with:
- [Sentence Transformers](https://www.sbert.net/)
- [Qdrant](https://qdrant.tech)
- [Streamlit](https://streamlit.io)
- [Scikit-learn](https://scikit-learn.org)
- Kaggle dataset: [IMDB Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
""")
