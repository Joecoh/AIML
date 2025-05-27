from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, random_state=0)
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)
print("Cluster centers:\n", kmeans.cluster_centers_)
