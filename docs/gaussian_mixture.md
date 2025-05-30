
**示例代码**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 1. 生成混合高斯分布数据
def generate_data(n_samples=1000):
    # 定义三个高斯分布的参数
    means = [
        [2, 2],
        [8, 3],
        [3, 8]
    ]
    covs = [
        [[1, 0.3], [0.3, 1]],
        [[1, -0.5], [-0.5, 1]],
        [[0.8, 0], [0, 0.8]]
    ]
    weights = [0.3, 0.5, 0.2]
    
    # 生成数据
    data = []
    for _ in range(n_samples):
        # 按权重选择高斯分量
        k = np.random.choice(len(weights), p=weights)
        x = np.random.multivariate_normal(means[k], covs[k])
        data.append(x)
    return np.array(data)

# 2. 高斯混合模型实现
class GMM:
    def __init__(self, n_components=3, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, X):
        # 初始化参数
        n_samples, n_features = X.shape
        
        # 随机初始化均值和协方差
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covs = [np.eye(n_features)] * self.n_components
        self.weights = np.ones(self.n_components) / self.n_components
        
        # EM算法
        prev_log_likelihood = None
        for iter in range(self.max_iter):
            # E-step：计算后验概率
            responsibilities = self._e_step(X)
            
            # M-step：更新参数
            self._m_step(X, responsibilities)
            
            # 计算对数似然
            log_likelihood = self._compute_log_likelihood(X)
            
            # 收敛判断
            if prev_log_likelihood is not None and \
                abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = log_likelihood
    
    def _e_step(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # 计算每个分量的概率密度
            rv = multivariate_normal(
                mean=self.means[k], 
                cov=self.covs[k],
                allow_singular=True
            )
            responsibilities[:, k] = self.weights[k] * rv.pdf(X)
            
        # 归一化得到后验概率
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        
        # 更新权重
        self.weights = responsibilities.sum(axis=0) / n_samples
        
        # 更新均值和协方差
        for k in range(self.n_components):
            resp_k = responsibilities[:, k].reshape(-1, 1)
            
            # 均值
            self.means[k] = (resp_k * X).sum(axis=0) / resp_k.sum()
            
            # 协方差
            diff = X - self.means[k]
            self.covs[k] = (resp_k * diff.T @ diff) / resp_k.sum()
            
    def _compute_log_likelihood(self, X):
        log_likelihood = 0
        for k in range(self.n_components):
            rv = multivariate_normal(
                mean=self.means[k], 
                cov=self.covs[k],
                allow_singular=True
            )
            log_likelihood += self.weights[k] * rv.pdf(X)
        return np.log(log_likelihood).sum()
    
    def predict(self, X):
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

# 3. 使用示例
if __name__ == "__main__":
    # 生成数据
    np.random.seed(42)
    X = generate_data(1000)
    
    # 训练模型
    gmm = GMM(n_components=3, max_iter=100)
    gmm.fit(X)
    
    # 预测聚类结果
    labels = gmm.predict(X)
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=5, cmap='viridis')
    plt.scatter(np.array(gmm.means)[:, 0], 
                np.array(gmm.means)[:, 1], 
                c='red', s=100, marker='x')
    plt.title("GMM Clustering Result")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
```

