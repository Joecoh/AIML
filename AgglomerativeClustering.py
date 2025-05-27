from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, random_state=0)
agg = AgglomerativeClustering(n_clusters=4)
labels = agg.fit_predict(X)
print(labels)
