**代码示例**

```python
import tensorflow as tf
import os
import numpy as np

class RL_QG_agent:
    def __init__(self):
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reversi")
        os.makedirs(self.model_dir, exist_ok=True)
        self.sess = None
        self.saver = None
        self.input_states = None
        self.Q_values = None
        self.init_model()  # 初始化模型结构

    def init_model(self):
        # 定义神经网络结构
        self.sess = tf.Session()
        # 输入为8x8棋盘，3个通道（当前玩家、对手、可行位置）
        self.input_states = tf.placeholder(tf.float32, shape=[None, 8, 8, 3], name="input_states")
        # 卷积层
        conv1 = tf.layers.conv2d(inputs=self.input_states, filters=32, kernel_size=3, padding="same", activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=3, padding="same", activation=tf.nn.relu)
        # 扁平化
        flat = tf.layers.flatten(conv2)
        # 全连接层
        dense = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu)
        # 输出层（66个动作：0-63为位置，64为认输，65为pass）
        self.Q_values = tf.layers.dense(inputs=dense, units=66, name="q_values")
        # 初始化变量和Saver
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def place(self, state, enables):
        # 预处理状态（3x8x8 -> 8x8x3）
        state_array = np.array(state)
        state_transposed = np.transpose(state_array, (1, 2, 0))  # 转换轴顺序
        state_input = state_transposed[np.newaxis, ...].astype(np.float32)
        
        # 获取所有动作的Q值
        q_vals = self.sess.run(self.Q_values, feed_dict={self.input_states: state_input})
        
        # 过滤合法动作的Q值
        legal_actions = enables
        if not legal_actions:
            return 8**2 + 1  # 返回pass动作
        
        # 选择最大Q值的合法动作
        legal_q = q_vals[0][legal_actions]
        if np.all(legal_q == 0):
            return np.random.choice(legal_actions)  # 随机选择
        
        max_q = np.max(legal_q)
        best_actions = [a for a in legal_actions if q_vals[0][a] == max_q]
        return np.random.choice(best_actions)

    def save_model(self):
        self.saver.save(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    def load_model(self):
        self.saver.restore(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))
```

