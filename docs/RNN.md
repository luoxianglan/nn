# 循环神经网络

**​​目录结构**​​  
poetry_gen/  
├── data/  
│   └── poems.txt  
├── models/  
│   ├── tf_rnn.py         # TensorFlow实现  
│   └── pytorch_rnn.py     # PyTorch实现  
├── utils/  
│   └── data_utils.py      # 数据处理  
└── train.py               # 训练入口  


```python
#TensorFlow实现：
class myRNNModel(keras.Model):
    def __init__(self, w2id):
        super().__init__()
        self.v_sz = len(w2id)
        self.embed_layer = tf.keras.layers.Embedding(self.v_sz, 64, batch_input_shape=[None, None])
        self.rnncell = tf.keras.layers.SimpleRNNCell(128)
        self.rnn_layer = tf.keras.layers.RNN(self.rnncell, return_sequences=True, return_state=True)  # 补全return_state
        self.dense = tf.keras.layers.Dense(self.v_sz, activation='softmax')  # 补全softmax激活

    @tf.function
    def call(self, inp_ids):
        x = self.embed_layer(inp_ids)
        outputs, final_state = self.rnn_layer(x)  # 补全获取最终状态
        logits = self.dense(outputs)
        return logits, final_state  # 返回logits和状态

    @tf.function
    def get_next_token(self, x, state):
        x = tf.expand_dims(x, 0)  # 转换为[1,1]形状
        x_emb = self.embed_layer(x)
        h, new_state = self.rnncell(x_emb, state)
        logits = self.dense(h)
        token = tf.argmax(logits, axis=-1)
        return token[0][0], new_state  # 返回单个token和更新后的状态
        

#训练部分补充：
def train_one_step(model, optimizer, x, y, seqlen):
    with tf.GradientTape() as tape:
        logits, _ = model(x)  # 直接获取logits和状态
        loss = compute_loss(logits, y, seqlen)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# 生成时需维护状态
def gen_sentence(model, start_token, end_token, max_len=50):
    model.load_weights('rnn_model.h5')
    state = model.rnn_layer.cell.get_initial_state(batch_size=1, dtype=tf.float32)
    cur_token = tf.constant([[word2id[start_token]]], dtype=tf.int32)
    
    collect = []
    for _ in range(max_len):
        logits, state = model(cur_token, state)  # 传入状态
        token = tf.argmax(logits, axis=-1)[0][0]
        collect.append(token)
        if token == word2id[end_token]:
            break
    return [id2word[t] for t in collect if t != word2id[end_token]]
    
#补全代码（rnn_lstm.py）​​：
class RNN_model(nn.Module):
    def __init__(self, batch_sz, vocab_len, word_embedding, embedding_dim, lstm_hidden_dim):
        super().__init__()
        self.word_embedding = word_embedding
        self.rnn_lstm = nn.LSTM(  # 补全参数
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True  # 修改为batch_first=True
        )
        self.fc = nn.Linear(lstm_hidden_dim, vocab_len)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        x = self.word_embedding(x)
        output, hidden = self.rnn_lstm(x, hidden)  # 补全hidden参数
        output = self.fc(output.reshape(output.size(0)*output.size(1), output.size(2)))
        return self.softmax(output), hidden

    def init_hidden(self, batch_size):
        return (
            torch.zeros(2, batch_size, self.rnn_lstm.hidden_size),
            torch.zeros(2, batch_size, self.rnn_lstm.hidden_size)
        )
        
 ​#​训练循环缺少状态管理：
 def train_one_step(model, optimizer, x, y):
    optimizer.zero_grad()
    hidden = model.init_hidden(x.size(0))
    output, _ = model(x, hidden)
    loss = F.nll_loss(output.view(-1, output.size(2)), y.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

# 生成诗歌示例
def generate_poem(model, start_word, max_length=20):
    model.eval()
    with torch.no_grad():
        hidden = model.init_hidden(1)
        input_tensor = torch.tensor([[word2idx[start_word]]], dtype=torch.long)
        poem = [start_word]
        
        for _ in range(max_length):
            output, hidden = model(input_tensor, hidden)
            topv, topi = output.topk(1)
            next_word = topi.item()
            poem.append(idx2word[next_word])
            input_tensor = torch.tensor([[next_word]], dtype=torch.long)
            if next_word == word2idx['EOS']:
                break
    return ' '.join(poem)
    
#缺失部分​​：未过滤非法字符，词汇表构建错误
#​​补全代码​​：
def preprocess_text(text):
    text = text.replace('\n', ' ').replace('\r', '')
    text = re.sub(r'[^\u4e00-\u9fff]', ' ', text)  # 仅保留中文
    return text

# 构建词汇表时添加特殊符号
special_tokens = {'PAD': 0, 'UNK': 1, 'BOS': 2, 'EOS': 3}
word_counts = Counter(all_words)
for token in special_tokens:
    word_counts[token] = 1e9  # 确保特殊符号优先
sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
words, _ = zip(*sorted_words)
word2idx = {word: idx for idx, word in enumerate(words)}

​#​生成控制补全代码：
def generate_with_temperature(model, start_token, temperature=0.8, max_len=50):
    state = model.rnn_layer.cell.get_initial_state(batch_size=1, dtype=tf.float32)
    cur_token = tf.constant([[word2id[start_token]]])
    
    collect = []
    for _ in range(max_len):
        logits, state = model.get_next_token(cur_token, state)
        logits = logits / temperature  # 应用温度
        token = tf.random.categorical(logits, 1)[0][0]
        collect.append(token.numpy()[0])
        if token == word2id['eos']:
            break
        cur_token = tf.constant([[token]])
    return [id2word[t] for t in collect if t != word2id['eos']]
    
#添加Beam Search​​（PyTorch示例）：
def beam_search(model, start_token, beam_size=5, max_len=20):
    model.eval()
    with torch.no_grad():
        sequences = [[word2idx[start_token]]]
        scores = [0.0]
        
        for _ in range(max_len):
            candidates = []
            for seq, score in zip(sequences, scores):
                input_tensor = torch.tensor([seq], dtype=torch.long)
                hidden = model.init_hidden(1)
                output, _ = model(input_tensor, hidden)
                log_probs = F.log_softmax(output[0, -1], dim=0)
                
                topk = log_probs.topk(beam_size)
                for lprob, idx in zip(topk.values, topk.indices):
                    new_seq = seq + [idx.item()]
                    new_score = score + lprob.item()
                    candidates.append( (new_seq, new_score) )
            
            sequences, scores = zip(*sorted(candidates, key=lambda x: -x[1])[:beam_size])
        
        return [id2word[idx] for idx in sequences[0] if idx != word2idx['EOS']]
        
#生成诗歌​​：
from models.tf_rnn import generate_with_temperature
print(generate_with_temperature('山', temperature=0.7))
```          

