
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,davies_bouldin_score
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv("ted_talks_en.csv") 
df.dropna(subset=['title', 'topics','description'], inplace=True)

random_state=55688
n_clusters=5

def show_result(name, result,title_num=5,n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df['result'] = kmeans.fit_predict(result)

    print("method",name)
    print("num of clusters:",n_clusters)
    print("Silhouette Score:", silhouette_score(result, df['result']))
    print("Davies-Bouldin Index(lower better):", davies_bouldin_score(result, df['result']))

    for i in range(n_clusters):
        cluster_titles = df.loc[ df['result'] == i, 'title'].sample(title_num, random_state=random_state).values
        print("cluster:",i)
        for title in cluster_titles:
            print(title)
        print()


cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(df['description'])

lda = LatentDirichletAllocation(n_components=n_clusters, random_state=random_state)
lda_result = lda.fit_transform(count_matrix)
show_result("lda",lda_result,n_clusters=n_clusters)

print("\n\n\n")

transformer = SentenceTransformer('all-MiniLM-L6-v2')
transformer_result = transformer.encode(df['description'])
show_result("transformer",transformer_result,n_clusters=n_clusters)

#reference: slides
#reference: https://medium.com/@dingusagar/text-clustering-using-sentence-embeddings-abcb6048fc36
#reference: https://www.geeksforgeeks.org/machine-learning/clustering-performance-evaluation-in-scikit-learn/
