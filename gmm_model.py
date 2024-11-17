import numpy as np
from sklearn.cluster import KMeans

class GMM:
    def __init__(self, n_components=1, max_iter=200, tol=1e-6):
        """
        初始化高斯混合模型
        Parameters:
            n_components (int): 高斯分布的数量（即簇的数量）。
            max_iter (int): 最大迭代次数。
            tol (float): 收敛阈值，判断模型是否收敛。
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None  # 混合系数
        self.means = None  # 高斯分布的均值
        self.covariances = None  # 高斯分布的协方差矩阵
    
    def fit(self, X):
        """
        使用 EM 算法训练 GMM 模型
        Parameters:
            X (numpy.ndarray): 输入数据，形状为 (n_samples, n_features)。
        """
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.ones(self.n_components) / self.n_components
        # 使用 K-Means 初始化均值
        self.means = KMeans(n_clusters=self.n_components, random_state=42).fit(X).cluster_centers_
        self.covariances = np.array([np.cov(X.T) + np.eye(n_features) * 1e-6 for _ in range(self.n_components)])

        log_likelihood = 0
        
        for iteration in range(self.max_iter):
            # E-step: 计算每个样本属于每个分布的责任值
            responsibilities = self._e_step(X)
            
            # M-step: 更新权重、均值和协方差
            self._m_step(X, responsibilities)
            
            # 检查是否收敛
            new_log_likelihood = self._compute_log_likelihood(X)
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                print(f"Converged at iteration {iteration}")
                break
            log_likelihood = new_log_likelihood
    
    def predict(self, X):
        """
        预测样本属于的簇标签
        Parameters:
            X (numpy.ndarray): 输入数据，形状为 (n_samples, n_features)。
        Returns:
            numpy.ndarray: 每个样本的簇标签，形状为 (n_samples,)。
        """
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def get_parameters(self):
        """
        获取模型参数
        Returns:
            tuple: 包括均值 (numpy.ndarray) 和协方差矩阵 (numpy.ndarray)。
        """
        return self.means, self.covariances
    
    def _e_step(self, X):
        """
        期望步 (E-step): 计算责任值（即每个样本属于每个分布的概率）。
        Parameters:
            X (numpy.ndarray): 输入数据。
        Returns:
            numpy.ndarray: 责任值矩阵，形状为 (n_samples, n_components)。
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * self._multivariate_gaussian(X, self.means[k], self.covariances[k])
        responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
        responsibilities_sum[responsibilities_sum == 0] = 1e-6
        return responsibilities / responsibilities_sum
    
    def _m_step(self, X, responsibilities):
        """
        最大化步 (M-step): 根据责任值更新模型参数。
        Parameters:
            X (numpy.ndarray): 输入数据。
            responsibilities (numpy.ndarray): 责任值矩阵。
        """
        n_samples, n_features = X.shape
        effective_n = responsibilities.sum(axis=0)  # 每个分布的有效样本数量
        
        # 更新混合系数
        self.weights = effective_n / n_samples
        
        # 更新均值
        self.means = np.dot(responsibilities.T, X) / effective_n[:, np.newaxis]
        
        # 更新协方差矩阵
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = np.dot((responsibilities[:, k][:, np.newaxis] * diff).T, diff) / effective_n[k]
            if effective_n[k] < 1e-6:  # 重新初始化空簇
                self.means[k] = X[np.random.choice(X.shape[0])]
                self.covariances[k] = np.cov(X.T) + np.eye(X.shape[1]) * 1e-6
    
    def _compute_log_likelihood(self, X):
        """
        计算对数似然，用于判断收敛
        Parameters:
            X (numpy.ndarray): 输入数据。
        Returns:
            float: 对数似然值。
        """
        log_likelihood = 0
        for k in range(self.n_components):
            log_likelihood += self.weights[k] * self._multivariate_gaussian(X, self.means[k], self.covariances[k])
        return np.sum(np.log(log_likelihood + 1e-6))  # 防止 log(0)
    
    @staticmethod
    def _multivariate_gaussian(X, mean, covariance):
        """
        计算多元高斯分布的概率密度函数值
        Parameters:
            X (numpy.ndarray): 输入数据。
            mean (numpy.ndarray): 均值向量。
            covariance (numpy.ndarray): 协方差矩阵。
        Returns:
            numpy.ndarray: 每个样本的概率密度值。
        """
        n_features = X.shape[1]
        # 添加正则项以确保协方差矩阵正定
        covariance += np.eye(covariance.shape[0]) * 1e-6
        covariance_det = np.linalg.det(covariance)
        if covariance_det <= 0:
            covariance_det = 1e-6  # 避免数值错误
        covariance_inv = np.linalg.inv(covariance)
        diff = X - mean

        exponent = np.einsum('ij,jk,ik->i', diff, covariance_inv, diff)  # 向量化计算 (x-μ)Σ⁻¹(x-μ)
        return np.exp(-0.5 * exponent) / np.sqrt((2 * np.pi) ** n_features * covariance_det)