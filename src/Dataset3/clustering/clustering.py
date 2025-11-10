
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("ted_talks_en.csv") 
df.dropna(subset=['title', 'topics','description'], inplace=True)

n_clusters = 10  

def show_result( result,top_count_num=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['result'] = kmeans.fit_predict(result)

    cluster_centers = np.vstack([
        np.asarray(result[df['result'].values == i].mean(axis=0)).ravel()
        for i in range(n_clusters)
    ])

    for i in range(n_clusters):
        cluster_mask = df['result'].values == i
        cluster_vecs = result[cluster_mask]

        if not isinstance(cluster_vecs, np.ndarray):
            cluster_vecs = cluster_vecs.toarray()

        center_vec = cluster_centers[i].reshape(1, -1)
        sims = cosine_similarity(cluster_vecs, center_vec).flatten()
        top_idx = sims.argsort()[-top_count_num:][::-1]

        cluster_titles = df.loc[cluster_mask, 'title'].iloc[top_idx].values

        print(f"cluster:{i}")
        for t in cluster_titles:
            print(f" - {t}")

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_result = tfidf_vectorizer.fit_transform(df['description'])
show_result(tfidf_result)


transformer = SentenceTransformer('all-MiniLM-L6-v2')
transformer_result = transformer.encode(df['description'])
show_result(transformer_result)

