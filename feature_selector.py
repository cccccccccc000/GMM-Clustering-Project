def select_features(X, selected_features=None):
    """
    从数据集中选择特定特征。
    
    Parameters:
        X (pd.DataFrame): 原始数据集。
        selected_features (list): 要选择的特征列表。如果为 None，则返回所有特征。
        
    Returns:
        pd.DataFrame: 包含选定特征的数据集。
    """
    if selected_features is None:
        return X  # 返回所有特征
    return X[selected_features]
