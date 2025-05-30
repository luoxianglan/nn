**代码实现**

```python
    def __init__(self):
        super(mySeq2SeqModel, self).__init__()
        self.v_sz = 27
        self.hidden = 128
        self.embed_layer = tf.keras.layers.Embedding(self.v_sz, 64, 
                                                    batch_input_shape=[None, None])
        
        self.encoder_cell = tf.keras.layers.SimpleRNNCell(self.hidden)
        self.decoder_cell = tf.keras.layers.SimpleRNNCell(self.hidden)
        
        self.encoder = tf.keras.layers.RNN(self.encoder_cell, 
                                           return_sequences=True, return_state=True)
        self.decoder = tf.keras.layers.RNN(self.decoder_cell, 
                                           return_sequences=True, return_state=True)
        self.dense_attn = tf.keras.layers.Dense(self.hidden)
        self.dense = tf.keras.layers.Dense(self.v_sz)
        
    @tf.function
    def call(self, enc_ids, dec_ids):
        #完成带attention机制的 sequence2sequence 模型的搭建
        # 编码阶段
        enc_emb = self.embed_layer(enc_ids)
        enc_outputs, enc_state = self.encoder(enc_emb)  # enc_outputs: (batch, seq_len, hidden)
        
        # 解码阶段
        dec_emb = self.embed_layer(dec_ids)
        dec_outputs, _ = self.decoder(dec_emb, initial_state=enc_state)
        
        # 注意力机制
        attn_scores = tf.matmul(dec_outputs, enc_outputs, transpose_b=True)  # (batch, dec_len, enc_len)
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)  # (batch, dec_len, enc_len)
        context = tf.matmul(attn_weights, enc_outputs)  # (batch, dec_len, hidden)
        
        # 结合上下文
        combined = tf.concat([dec_outputs, context], axis=-1)  # (batch, dec_len, hidden*2)
        attn_output = self.dense_attn(combined)  # (batch, dec_len, dense_units)
        logits = self.dense(attn_output)  # (batch, dec_len, vocab_size)
        
        return logits
    
    @tf.function
    def encode(self, enc_ids):
        enc_emb = self.embed_layer(enc_ids)
        enc_outputs, enc_state = self.encoder(enc_emb)
        return enc_outputs, enc_state  # 返回全部编码输出和最后状态
    
    def get_next_token(self, x, state, enc_outputs):
        #实现单步解码逻辑（带注意力）
        x_emb = self.embed_layer(x)  # (batch, emb_dim)
        
        # RNN解码步骤
        output, new_state = self.decoder_cell(x_emb, state)
        
        # 计算注意力权重
        attn_scores = tf.matmul(output, enc_outputs, transpose_b=True)  # (batch, 1, enc_len)
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)  # (batch, 1, enc_len)
        context = tf.matmul(attn_weights, enc_outputs)  # (batch, 1, hidden)
        
        # 结合上下文
        combined = tf.concat([output, tf.squeeze(context, axis=1)], axis=-1)  # (batch, hidden*2)
        attn_logits = self.dense_attn(combined)  # (batch, hidden_attn)
        logits = self.dense(attn_logits)  # (batch, vocab_size)
        
        predicted_id = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return predicted_id, new_state
```

