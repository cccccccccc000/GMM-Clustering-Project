import pandas as pd
from data_loader import load_dataset
from feature_selector import select_features
from gmm_model import GMM
from cluster_metrics import find_optimal_clusters
from visualization import plot_clusters
from sklearn.metrics import adjusted_rand_score, accuracy_score

def main():
    # 加载数据
    X, y, metadata = load_dataset(dataset_id=109)
    
    # 确保 y 是一个 Series
    if isinstance(y, pd.DataFrame):  # 如果 y 是 DataFrame，提取其中的第一列
        y = y.iloc[:, 0]
    
    # 输出数据集的 target 信息
    unique_targets = y.unique()  # 获取唯一目标类别值
    print(f"The dataset has {len(unique_targets)} unique targets: {list(unique_targets)}")
    
    # 动态选择特征
    # selected_features = ['sepal length', 'sepal width']  # 可修改为动态输入
    selected_features = None
    X_selected = select_features(X, selected_features).values  # 转为 NumPy 数组
    
    # 使用数据集中的类别数量确定聚类数量
    n_clusters = len(unique_targets)
    print(f"Number of clusters determined by target classes: {n_clusters}")
    
    # 自动选择最佳聚类数量（可选）
    # optimal_clusters, bic_scores, sil_scores = find_optimal_clusters(X_selected, max_clusters=20)
    # print(f"Optimal Number of Clusters: {optimal_clusters}")
    
    # 使用类别数量作为聚类数量训练 GMM
    gmm = GMM(n_components=n_clusters)
    gmm.fit(X_selected)
    labels = gmm.predict(X_selected)
    
    # 对比聚类结果与真实标签

    # 评价聚类结果
    ari = adjusted_rand_score(y, labels)
    print(f"Adjusted Rand Index (ARI): {ari}")
    # print(y)
    # print(labels)
    
    # 可选：计算准确率（如果类别数相同且类别顺序一致）
    # try:
        # accuracy = accuracy_score(y, labels)
        # print(f"Accuracy: {accuracy}")
    # except ValueError:
        # print("Accuracy calculation is not applicable because the label assignments differ.")

    # 获取参数并可视化
    means, covariances = gmm.get_parameters()
    plot_clusters(X_selected, labels, means, covariances)

if __name__ == "__main__":
    main()