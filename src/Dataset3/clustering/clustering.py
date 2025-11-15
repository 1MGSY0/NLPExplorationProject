
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv("ted_talks_en.csv") 
df.dropna(subset=['title', 'topics','description'], inplace=True)

def show_result( result,top_count_num=10,n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['result'] = kmeans.fit_predict(result)

    score = silhouette_score(result, df['result'])
    print("num of clusters:",n_clusters)
    print("Silhouette Score:", score)

    for i in range(n_clusters):
        cluster_titles = df.loc[ df['result'] == i, 'title'].iloc[0:top_count_num].values
        print(f"cluster:{i}")
        for t in cluster_titles:
            print(f" - {t}")


cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(df['description'])

lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda_result = lda.fit_transform(count_matrix)
show_result(lda_result,n_clusters=15)


transformer = SentenceTransformer('all-MiniLM-L6-v2')
transformer_result = transformer.encode(df['description'])
show_result(transformer_result,n_clusters=15)

