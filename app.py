import streamlit as st
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD, NMF

# =============================
# Page Config
# =============================
st.set_page_config(page_title=" Movie Recommender", layout="wide")

# =============================
# Load Data
# =============================
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies, ratings = load_data()

# =============================
# Content-Based (Embeddings)
# =============================
@st.cache_resource
def load_embeddings():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    movies['content'] = movies['title'] + " " + movies['genres'].str.replace("|", " ")
    embeddings = model.encode(movies['content'].tolist(), show_progress_bar=False)
    return np.array(embeddings).astype('float32')

embeddings = load_embeddings()

# =============================
# FAISS
# =============================
@st.cache_resource
def build_faiss(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

faiss_index = build_faiss(embeddings)

# =============================
# User-Movie Matrix
# =============================
@st.cache_resource
def build_matrix():
    return ratings.pivot_table(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)

user_movie = build_matrix()

# =============================
# KNN
# =============================
@st.cache_resource
def train_knn():
    knn = NearestNeighbors(metric='cosine')
    knn.fit(user_movie)
    return knn

knn_model = train_knn()

# =============================
# SVD (Reconstruction)
# =============================
@st.cache_resource
def train_svd():
    svd = TruncatedSVD(n_components=50, random_state=42)
    user_latent = svd.fit_transform(user_movie)
    item_latent = svd.components_
    return np.dot(user_latent, item_latent)

svd_matrix = train_svd()

# =============================
# NMF (Reconstruction)
# =============================
@st.cache_resource
def train_nmf():
    nmf = NMF(n_components=50, init='random', random_state=42, max_iter=200)
    user_latent = nmf.fit_transform(user_movie)
    item_latent = nmf.components_
    return np.dot(user_latent, item_latent)

nmf_matrix = train_nmf()

# =============================
# Recommendation Functions
# =============================

def content_recommend(idx, top_k):
    query = embeddings[idx].reshape(1, -1)
    _, indices = faiss_index.search(query, k=top_k + 1)

    recs = movies.iloc[indices[0]]
    recs = recs[recs.index != idx]

    return recs.head(top_k).to_dict('records')


def knn_recommend(user_id, top_k):
    if user_id not in user_movie.index:
        return []

    user_vec = user_movie.loc[user_id].values.reshape(1, -1)
    distances, indices = knn_model.kneighbors(user_vec, n_neighbors=top_k + 1)

    similar_users = user_movie.iloc[indices[0][1:]]
    weights = 1 - distances[0][1:]

    scores = np.average(similar_users, axis=0, weights=weights)

    top_indices = np.argsort(scores)[::-1][:top_k]
    movie_ids = user_movie.columns[top_indices]

    return movies[movies['movieId'].isin(movie_ids)].to_dict('records')


def svd_recommend(user_id, top_k):
    if user_id not in user_movie.index:
        return []

    user_idx = list(user_movie.index).index(user_id)
    scores = svd_matrix[user_idx]

    top_indices = np.argsort(scores)[::-1][:top_k]
    movie_ids = user_movie.columns[top_indices]

    return movies[movies['movieId'].isin(movie_ids)].to_dict('records')


def nmf_recommend(user_id, top_k):
    if user_id not in user_movie.index:
        return []

    user_idx = list(user_movie.index).index(user_id)
    scores = nmf_matrix[user_idx]

    top_indices = np.argsort(scores)[::-1][:top_k]
    movie_ids = user_movie.columns[top_indices]

    return movies[movies['movieId'].isin(movie_ids)].to_dict('records')


def collaborative_recommend(user_id, top_k):
    knn_res = knn_recommend(user_id, top_k)
    svd_res = svd_recommend(user_id, top_k)
    nmf_res = nmf_recommend(user_id, top_k)

    combined = knn_res + svd_res + nmf_res

    seen = set()
    result = []

    for m in combined:
        if m['movieId'] not in seen:
            result.append(m)
            seen.add(m['movieId'])

    return result[:top_k]


def hybrid_recommend(idx, user_id, top_k):
    content = content_recommend(idx, top_k)
    collab = collaborative_recommend(user_id, top_k)

    combined = content + collab

    seen = set()
    result = []

    for m in combined:
        if m['movieId'] not in seen:
            result.append(m)
            seen.add(m['movieId'])

    return result[:top_k]


# =============================
# SIDEBAR
# =============================
st.sidebar.title(" Navigation")
page = st.sidebar.radio("Go to", [" Home", " Recommendation", " EDA", " About"])

# =============================
# HOME
# =============================
if page == " Home":
    st.title(" Movie Recommendation System")

    st.markdown("##  Project Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ###  Content-Based
        • NLP (Sentence Transformers)  
        • Movie similarity (title + genres)  
        """)

        st.markdown("""
        ###  Collaborative Filtering
        • Based on user ratings  
        • Learns user preferences  
        """)

    with col2:
        st.markdown("""
        ###  Hybrid Model
        • Combines both methods  
        • Better recommendations  
        """)

        st.markdown("""
        ###  FAISS
        • Fast similarity search  
        """)

    st.success(" Go to Recommendation tab to try!")

# =============================
# RECOMMENDATION
# =============================
elif page == " Recommendation":

    st.title(" Get Recommendations")

    movie_list = movies['title'].values
    selected_movie = st.selectbox("Select Movie", movie_list)

    idx = movies[movies['title'] == selected_movie].index[0]

    user_id = st.number_input("Enter User ID", 1, int(ratings['userId'].max()), 1)
    top_k = st.slider("Top K", 3, 10, 5)

    method = st.radio(
        "Choose Method",
        ["Content-Based", "Collaborative", "Hybrid (Best)"]
    )

    if st.button("Recommend"):

        if method == "Content-Based":
            recs = content_recommend(idx, top_k)

        elif method == "Collaborative":
            recs = collaborative_recommend(user_id, top_k)

        else:
            recs = hybrid_recommend(idx, user_id, top_k)

        st.subheader(" Recommended Movies")
        cols = st.columns(3)

        for i, movie in enumerate(recs):
            with cols[i % 3]:
                movie_ratings = ratings[ratings['movieId'] == movie['movieId']]['rating']
                avg_rating = round(movie_ratings.mean(), 2) if not movie_ratings.empty else "N/A"

                st.markdown(f"""
                <div style="background-color:#1f1f1f;
                            padding:15px;
                            border-radius:10px;
                            margin:10px;
                            text-align:center;
                            color:white;">
                    🎬 <b>{movie['title']}</b><br>
                    ⭐ Rating: {avg_rating}
                </div>
                """, unsafe_allow_html=True)

# =============================
# EDA
# =============================
elif page == " EDA":
    st.title(" Data Analysis")

    st.subheader("Ratings Distribution")
    st.bar_chart(ratings['rating'].value_counts())

    st.subheader("Top Rated Movies")
    avg = ratings.groupby('movieId')['rating'].mean().reset_index()
    top = avg.merge(movies, on='movieId').sort_values(by='rating', ascending=False)
    st.dataframe(top[['title','rating']].head(10))

    st.subheader("Most Watched Movies")
    watch = ratings['movieId'].value_counts().reset_index()
    watch.columns = ['movieId','count']
    most = watch.merge(movies, on='movieId')

    st.dataframe(most[['title','count']].head(10))

    st.subheader("Top 10 Most Watched (Chart)")
    chart = most[['title','count']].head(10).set_index('title')
    st.bar_chart(chart)

# =============================
# ABOUT
# =============================
elif page == " About":
    st.title(" About Project")

    st.markdown("""
    ##  Movie Recommendation System

    This project is an **AI-powered movie recommendation system** built using Machine Learning, NLP, and similarity techniques.

    ---

    ##  Technologies Used

    ✔ **Natural Language Processing (NLP)**  
    • Sentence Transformers (all-MiniLM-L6-v2)  
    • Converts movie data into embeddings  

    ✔ **Content-Based Filtering**  
    • Recommends movies based on similarity (title + genres)  

    ✔ **Collaborative Filtering**  
    • Uses user rating data  
    • Trained using **KNN, Truncated SVD, and NMF models**  

    ✔ **Hybrid Recommendation System**  
    • Combines content-based and collaborative filtering  

    ✔ **FAISS (Fast Similarity Search)**  
    • Efficient nearest neighbor search  

    ✔ **Streamlit**  
    • Interactive web application UI  

    ---

    ##  Dataset Used

    ✔ MovieLens Dataset  
    • movies.csv  
    • ratings.csv  

    ---

    ##  How It Works

    1. Convert movies into embeddings using NLP  
    2. Find similar movies using FAISS  
    3. Build user-movie rating matrix  
    4. Train ML models (KNN, SVD, NMF)  
    5. Generate recommendations using hybrid approach  

    ---

    ##  Objective

    To recommend movies based on:
    • User preferences  
    • Movie similarity  
    • Historical ratings  

    ---

     Built with  using AI & Machine Learning
    """)
# Footer
st.markdown("---")
st.markdown(" Built with  using AI & ML")