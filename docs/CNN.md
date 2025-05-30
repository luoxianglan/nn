**PyTorch版本 (CNN_pytorch.py) 补全部分：**

```python
# 第一个卷积层参数补全
self.conv1 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(2)
)

# 第二个卷积层参数补全
self.conv2 = nn.Sequential(
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(2)
)

# 展平操作补全
x = x.view(x.size(0), -1)  # 或 x = x.view(-1, 7*7*64)
TensorFlow版本 (CNN_tensorflow.py) 补全部分：

# 补全卷积和池化函数
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 补全两层卷积的8个参数（权重、偏置、激活、池化）
# 卷积层1
W_conv1 = weight_variable([7, 7, 1, 32])  # 7x7卷积核，输入通道1，输出通道32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 卷积层2
W_conv2 = weight_variable([5, 5, 32, 64])  # 5x5卷积核，输入通道32，输出通道64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

按AlexNet结构补全更多卷积层和全连接层：
class myConvModel(keras.Model):
    def __init__(self):
        super(myConvModel, self).__init__()
        # 第1层卷积（5x5核，32通道）
        self.conv1 = Conv2D(32, (5,5), activation='relu', padding='same')
        self.pool = MaxPooling2D((2,2), strides=2)
        # 第2层卷积（5x5核，64通道）
        self.conv2 = Conv2D(64, (5,5), activation='relu', padding='same')
        # 第3层卷积（3x3核，128通道）
        self.conv3 = Conv2D(128, (3,3), activation='relu', padding='same')
        # 全连接层
        self.flat = Flatten()
        self.fc1 = Dense(1024, activation='relu')
        self.dropout = Dropout(0.5)
        self.fc2 = Dense(10)
    def call(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.conv3(x)  # 添加更多层
        x = self.flat(x)
        x = self.dropout(self.fc1(x))
        return self.fc2(x)
```
