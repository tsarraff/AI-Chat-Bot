import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics
digits = load_digits()
data = scale(digits.data)
y = digits.target
#k = len(np.unique(y))
k = 10
samples, features = data.shape

def bench_k_means(esitmator, name, data):
    esitmator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name,esitmator.inertia_,
             metrics.homogeneity_score(y,esitmator.labels),
             metrics.completeness_score(y,esitmator.labels),
             metrics.v_measure_score(y,esitmator.labels),
             metrics.adjusted_rand_score(y,esitmator.labels),
             metrics.adjusted_mutual_info_score(y,esitmator.labels),
             metrics.silhouette_score(y,esitmator.labels, metric='euclidean')))
clf = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(clf, "1", data)