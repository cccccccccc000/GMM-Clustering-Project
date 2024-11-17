import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_dataset(dataset_id=53):
    iris = fetch_ucirepo(id=dataset_id)  # 动态加载数据集
    X = iris.data.features
    y = iris.data.targets  # 这是一个 NumPy 数组
    metadata = iris.metadata

    # 检查并处理缺失值（选择一种方式）
    # 方式 1: 删除含有缺失值的样本
    X = X.dropna()
    y = y[:len(X)]  # 保证 y 与 X 的长度对齐

    # 或者
    # 方式 2: 用均值填充缺失值
    # X = X.fillna(X.mean())
    
    return X, y, metadata
