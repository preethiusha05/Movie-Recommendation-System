# Movie Recommendation System

An AI-powered Movie Recommendation System built using **Machine Learning, NLP, and Vector Embeddings**.
This project recommends movies using **Content-Based, Collaborative, and Hybrid approaches**.

---

#  Project Description

This system:

* Uses **content-based filtering** with NLP embeddings
* Uses **collaborative filtering** based on user ratings
* Combines both into a **hybrid recommendation system**
* Uses **vector embeddings** for semantic similarity
* Built and deployed using **Streamlit**

---

#  Features

* Content-Based Recommendations (NLP + FAISS)
* Collaborative Filtering (KNN, SVD, NMF)
* Hybrid Recommendation System
* Fast similarity search using FAISS
* Interactive UI using Streamlit
*  Data visualization (EDA)

---

#  Technologies Used

*  Python
*  Streamlit (UI development)
*  Pandas, NumPy (data processing)
*  Scikit-learn (machine learning models)
*  Sentence Transformers (NLP embeddings)
*  FAISS (vector similarity search)
---

#  Dataset

* MovieLens Dataset

  * `movies.csv`
  * `ratings.csv`

---

#  Step-by-Step Working

## 1️ Data Loading

* Load movie and rating datasets using Pandas
* Clean and prepare data

---
## 2 Content-Based Filtering
 Combine title + genres
 Convert text into embeddings
 
## 3️ Vector Search (FAISS)
 Store embeddings in FAISS index
 Perform similarity search
 
## 4️ Collaborative Filtering
 Build user-movie matrix
 Apply KNN, SVD, NMF models

## 5️ Hybrid Recommendation
 Combine results
 Remove duplicates
 Return top movies

# Machine Learning Models Used
 ## 1. Content-Based Model (NLP + Embeddings)
 * Model: SentenceTransformer (all-MiniLM-L6-v2)
 * Converts movie text (title + genres) into dense vector embeddings
 * Used for semantic similarity search
 * FAISS is used to find nearest neighbors efficiently

 ## 2. K-Nearest Neighbors (KNN)
 * Type: Collaborative Filtering
 * Trained on: User-Movie Rating Matrix
 * Uses cosine similarity to find similar users
 * Recommends movies based on neighbor preferences

 ## 3. Truncated SVD (Matrix Factorization)
 * Reduces high-dimensional rating matrix into latent features
 * Learns hidden patterns in user preferences
 * Uses matrix reconstruction to predict ratings

 ## 4. Non-Negative Matrix Factorization (NMF)
 * Factorizes matrix into non-negative latent features
 * Captures user-item relationships
 * Used for predicting ratings and recommendations

 ## 5. Hybrid Recommendation System
 Combines:
 * Content-Based recommendations
 * Collaborative filtering (KNN + SVD + NMF)
 * Removes duplicates and improves recommendation quality

 ## Training Summary
 * Models are trained on MovieLens dataset
 * User-movie interactions are converted into a matrix
 * ML models learn:
 * User preferences
 * Movie similarities
 * Predictions are generated dynamically during runtime

##  Streamlit UI

* Sidebar navigation:

  * Home
  * Recommendation
  * EDA
  * About
  * User inputs:

  * Select movie
  * Enter user ID
  * Choose method
  * Display recommendations dynamically

---

#  How to Run the Project

## Step 1: Clone Repository

```bash
git clone https://github.com/preethiusha05/Movie-Recommendation-System
cd "C:\Movie_Recommend_Project"
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Run Application

```bash
streamlit run app.py
```

---

#  Project Structure

```
Movie-Recommender/
│
├── app.py
├── movies.csv
├── ratings.csv
├── requirements.txt
└── README.md
```

---

#  Example Output

* Select a movie
* Enter user ID
* Choose recommendation method
* Get top recommended movies with ratings

---

#  Objective

To recommend movies based on:

* User preferences
* Movie similarity
* Historical ratings

---



#  Conclusion

This project demonstrates how **Machine Learning + NLP + Vector Search** can be combined to build an intelligent recommendation system.

---

