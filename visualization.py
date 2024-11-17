import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from matplotlib.cm import get_cmap
import random

def plot_clusters(X, labels, means, covariances):
    """
    随机选择 6 个子图，在同一个窗口中显示数据点分布及对应的高斯分布等高线。
    
    Parameters:
        X (numpy.ndarray): 数据点，形状为 (n_samples, n_features)。
        labels (numpy.ndarray): 聚类标签。
        means (numpy.ndarray): 每个聚类的均值，形状为 (n_clusters, n_features)。
        covariances (numpy.ndarray): 每个聚类的协方差矩阵，形状为 (n_clusters, n_features, n_features)。
    """
    n_features = X.shape[1]
    feature_combinations = list(combinations(range(n_features), 2))  # 所有特征的两两组合
    n_clusters = len(np.unique(labels))
    
    # 随机选择 6 个特征组合
    selected_combinations = random.sample(feature_combinations, min(6, len(feature_combinations)))
    
    # 使用颜色映射，为每个聚类分配颜色
    cmap = get_cmap('tab10', n_clusters)  # 最多支持 10 个不同颜色，可扩展
    cluster_colors = [cmap(i) for i in range(n_clusters)]
    
    # 创建一个大窗口用于显示所有选中的子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2x3 网格布局
    axes = axes.flatten()  # 将轴对象展平成一维数组以便索引
    
    for idx, (f1, f2) in enumerate(selected_combinations):
        ax = axes[idx]
        ax.set_title(f"Feature {f1 + 1} vs Feature {f2 + 1}")
        
        for cluster_idx, (mean, covar) in enumerate(zip(means, covariances)):
            color = cluster_colors[cluster_idx]
            
            # 筛选属于该簇的点
            cluster_data = X[labels == cluster_idx]
            ax.scatter(
                cluster_data[:, f1], 
                cluster_data[:, f2], 
                color=color, 
                label=f'Cluster {cluster_idx + 1}', 
                alpha=0.6
            )
            
            # 绘制等高线
            x, y = np.meshgrid(
                np.linspace(X[:, f1].min(), X[:, f1].max(), 100),
                np.linspace(X[:, f2].min(), X[:, f2].max(), 100)
            )
            xy = np.column_stack([x.flat, y.flat])
            
            # 提取对应特征的协方差和均值
            mean_ij = mean[[f1, f2]]
            covar_ij = covar[np.ix_([f1, f2], [f1, f2])]
            
            z = np.exp(-0.5 * np.sum(np.dot((xy - mean_ij), np.linalg.inv(covar_ij)) * (xy - mean_ij), axis=1))
            z = z.reshape(x.shape)
            ax.contour(x, y, z, levels=5, colors=[color], alpha=0.8)
        
        ax.set_xlabel(f"Feature {f1 + 1}")
        ax.set_ylabel(f"Feature {f2 + 1}")
    
    # 隐藏未使用的子图
    for idx in range(len(selected_combinations), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1))  # 调整图例位置
    plt.show()
