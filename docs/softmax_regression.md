# logistic_regression-exercise.py 

填空一：Sigmoid交叉熵损失函数
```python
# 在 compute_loss 函数中补充交叉熵损失计算
# 输入label shape(N,), pred shape(N,)
losses = -label * tf.math.log(pred + epsilon) - (1 - label) * tf.math.log(1 - pred + epsilon)
```

# softmax_regression-exercise.py 

填空一：模型参数初始化
```python
class SoftmaxRegression():
    def __init__(self):
        # 权重 W 形状 [2, 3]，偏置 b 形状 [3]
        self.W = tf.Variable(
            initial_value=tf.random.uniform(shape=[2, 3], minval=-0.1, maxval=0.1),
            dtype=tf.float32
        )
        self.b = tf.Variable(
            initial_value=tf.zeros(shape=[3]),
            dtype=tf.float32
        )
```


填空二：Softmax交叉熵损失函数
```python
@tf.function
def compute_loss(pred, label):
    label = tf.one_hot(tf.cast(label, dtype=tf.int32), depth=3, dtype=tf.float32)
    # 限制预测值范围，防止数值不稳定
    pred = tf.clip_by_value(pred, epsilon, 1.0)
    # 计算交叉熵损失
    losses = -tf.reduce_sum(label * tf.math.log(pred), axis=1)
    loss = tf.reduce_mean(losses)
```

Logistic回归： 
```python 
#损失函数实现逻辑
losses = -y*log(p) - (1-y)*log(1-p)
#添加 epsilon 防止数值溢出。
#输出示例：
#loss: 0.5123   accuracy: 0.8900

#Softmax回归：
#参数初始化：输入特征维度为2，输出类别为3，故:
 W.shape=[2,3]，b.shape=[3]。
#损失函数实现逻辑：
losses = -Σ(y_true * log(y_pred))
#使用 tf.clip_by_value 限制预测值范围。
#输出示例：
#loss: 0.2104   accuracy: 0.9500
```


