from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def find_optimal_clusters(X, max_clusters=20):
    """使用 BIC 和轮廓分数选择最佳聚类数量"""
    optimal_clusters = None
    lowest_bic = float('inf')
    bic_scores = []
    silhouette_scores = []
    
    for n in range(2, max_clusters + 1):
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        
        labels = gmm.predict(X)
        sil_score = silhouette_score(X, labels)
        silhouette_scores.append(sil_score)
        
        if gmm.bic(X) < lowest_bic:
            lowest_bic = gmm.bic(X)
            optimal_clusters = n
    
    return optimal_clusters, bic_scores, silhouette_scores
