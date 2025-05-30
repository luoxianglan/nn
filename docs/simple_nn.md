# softmax_ce 函数
潜在问题：函数参数名为 x，但实际输入应为 logits（未归一化的原始输出），而非经过 softmax 后的概率。当前实现假设 x 已经是概率分布，与交叉熵的标准定义不符。交叉熵的正确实现应对 logits 进行 softmax 后再计算，或直接使用 log_softmax 优化数值稳定性。
```python
#需补充代码：
def softmax_ce(x, label):
    #实现 softmax 交叉熵loss函数，不允许用tf自带的softmax_cross_entropy函数
    x = tf.cast(x, tf.float32)
    logits = x  # 假设 x 是未归一化的 logits
    # 数值稳定版 log_softmax
    x_max = tf.reduce_max(logits, axis=-1, keepdims=True)
    log_softmax = logits - x_max - tf.math.log(tf.reduce_sum(tf.exp(logits - x_max), axis=-1, keepdims=True))
    # 计算交叉熵
    loss = -tf.reduce_mean(tf.reduce_sum(label * log_softmax, axis=1))
    return loss   

#测试用例修正：
# 原始测试代码错误地传入了 softmax 后的概率，应直接传入 logits
test_data = np.random.normal(size=[10, 5])
label = np.zeros_like(test_data)
label[np.arange(10), np.random.randint(0, 5, size=10)] = 1.0
# 自定义损失
custom_loss = softmax_ce(test_data, label)
# TensorFlow 原生损失
tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(label, test_data))
# 验证一致性
assert (custom_loss - tf_loss)**2 < 1e-6
```


# sigmoid_ce 函数
潜在问题：参数 x 命名易混淆，应为 logits（未经过 sigmoid 的输出），而非概率值。当前实现假设 x 是概率，导致计算逻辑错误。正确的交叉熵公式应直接操作 logits，避免数值不稳定。
```python
#需补充代码：
def sigmoid_ce(x, label):
    #实现 sigmoid 交叉熵loss函数，不允许用tf自带的sigmoid_cross_entropy函数
    x = tf.cast(x, tf.float32)
    # 使用 logits 直接计算交叉熵，避免数值不稳定
    loss = tf.reduce_mean(
        tf.maximum(x, 0) - x * label + 
        tf.math.log(1 + tf.exp(-tf.abs(x)))
    )
    return loss
```

# 测试用例修正：
```python
test_data = np.random.normal(size=[10])
label = np.random.randint(0, 2, 10).astype(test_data.dtype)
# 自定义损失（直接传入 logits）
custom_loss = sigmoid_ce(test_data, label)
# TensorFlow 原生损失
tf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(label, test_data))
# 验证一致性
assert (custom_loss - tf_loss)**2 < 1e-6
```

'tutorial_minst_fnn-numpy-exercise.py'
```python
#Matmul 类的 backward 方法
def backward(self, grad_y):
    x: shape(N, d)
    w: shape(d, d')
    grad_y: shape(N, d')
    x = self.mem['x']
    W = self.mem['W']
    
    #计算矩阵乘法的对应的梯度'''
    grad_x = np.dot(grad_y, W.T)      # 计算 x 的梯度：grad_y * W^T
    grad_W = np.dot(x.T, grad_y)      # 计算 W 的梯度：x^T * grad_y
    return grad_x, grad_W


 #Relu 类的 backward 方法
def backward(self, grad_y):
    grad_y: same shape as x
    x = self.mem['x']
    #计算relu 激活函数对应的梯度
    grad_x = grad_y * (x > 0)         # 梯度为 grad_y 在 x > 0 的位置保留，否则置零
    return grad_x
```


'tutorial_minst_fnn-tf2.0-exercise.py'
```python
#train_one_step 函数中的梯度更新部分需将手动更新参数的方式替换为使用优化器：
# 原始错误代码：
# for g, v in zip(grads, trainable_vars):
#     v.assign_sub(0.01*g)

# 修改为使用优化器：
optimizer.apply_gradients(zip(grads, trainable_vars))  # 使用 Adam 优化器自动更新参数
```


