**代码示例**

```python
# python: 2.7
# encoding: utf-8
import numpy as np

class RBM:
    #Restricted Boltzmann Machine.

    def __init__(self, n_hidden=2, n_observe=784):
        #Initialize model.
        self.n_hidden = n_hidden
        self.n_observe = n_observe
        # 初始化权重和偏置
        self.W = np.random.normal(0, 0.01, size=(n_observe, n_hidden))  # 权重矩阵
        self.b_h = np.zeros(n_hidden)  # 隐藏层偏置
        self.b_v = np.zeros(n_observe)  # 可见层偏置

    def _sigmoid(self, x):
        #Sigmoid激活函数
        return 1.0 / (1 + np.exp(-x))

    def _sample_binary(self, probs):
        #伯努利采样
        return np.random.binomial(1, probs)

    def train(self, data):
        #Train model using data.
        # 将数据展平为二维数组 [n_samples, n_observe]
        data_flat = data.reshape(data.shape[0], -1)
        n_samples = data_flat.shape[0]
        learning_rate = 0.1
        epochs = 10
        batch_size = 100

        for epoch in range(epochs):
            # 打乱数据顺序
            np.random.shuffle(data_flat)
            for i in range(0, n_samples, batch_size):
                batch = data_flat[i:i + batch_size]
                v0 = batch.astype(np.float64)  # 确保数据类型正确

                # 正相传播
                h0_prob = self._sigmoid(np.dot(v0, self.W) + self.b_h)
                h0_sample = self._sample_binary(h0_prob)

                # 负相传播（CD-1）
                v1_prob = self._sigmoid(np.dot(h0_sample, self.W.T) + self.b_v)
                v1_sample = self._sample_binary(v1_prob)
                h1_prob = self._sigmoid(np.dot(v1_sample, self.W) + self.b_h)

                # 计算梯度（对比散度）
                dW = np.dot(v0.T, h0_sample) - np.dot(v1_sample.T, h1_prob)
                db_v = np.sum(v0 - v1_sample, axis=0)
                db_h = np.sum(h0_sample - h1_prob, axis=0)

                # 更新参数（批量平均）
                self.W += learning_rate * dW / batch_size
                self.b_v += learning_rate * db_v / batch_size
                self.b_h += learning_rate * db_h / batch_size

    def sample(self):
        #Sample from trained model using Gibbs sampling.
        # 初始化随机可见层（伯努利分布）
        v = np.random.binomial(1, 0.5, self.n_observe)
        # 进行1000次Gibbs采样
        for _ in range(1000):
            h_prob = self._sigmoid(np.dot(v, self.W) + self.b_h)
            h_sample = self._sample_binary(h_prob)
            v_prob = self._sigmoid(np.dot(h_sample, self.W.T) + self.b_v)
            v = self._sample_binary(v_prob)
        return v.reshape(28, 28)  # 恢复图像形状

# 训练和生成样本的主程序
if __name__ == '__main__':
    # 加载二值化后的MNIST数据（需提前处理为0/1）
    mnist = np.load('mnist_bin.npy')  # 假设数据已预处理为60000x28x28的0/1矩阵
    n_imgs, n_rows, n_cols = mnist.shape
    img_size = n_rows * n_cols

    # 构建RBM模型（隐变量设为100个）
    rbm = RBM(n_hidden=100, n_observe=img_size)

    # 训练RBM
    rbm.train(mnist)

    # 生成样本
    generated_image = rbm.sample()
    print("Generated sample shape:", generated_image.shape)
```

