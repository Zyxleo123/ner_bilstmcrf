# PJ2: 基于LSTM-CRF的序列标注

> 赵一溪 21307130052

## 1. 代码

### 1.1 模型


1. 总架构

    **Bert-like model(as embedding) -> BiLSTM -> 字符到词语转换 -> Emit score映射层(BiLSTM hidden size -> #Label) -> CRF**

    1. Bert-like model：载入Huggingface的预训练模型，用于生成上下文有关的高质量词向量

    2. BiLSTM：和后面的所有层一样为fine tune的预测头。用于提取适合本次数据集的上下文有关的特征。之所以不去掉这一层，是因为否则需要把task-specific的特征提取放到Bert-like model中，但是bert-like model的参数量太大相较数据量，不适合fine-tuning。

    3. 字符到词语转换：用于将BiLSTM的输出映射到每个词语的输出。这一层将[<CLS>, subword_embedding_1_1, subword_embedding_1_2, subword_embedding_2_1, ..., subword_embedding_n_3, <SEP>, <PAD>, <PAD>, ..., <PAD>]映射到[word_embedding_1, word_embedding_2, ..., word_embedding_n]。目的是使得label和feature的长度对齐，这样才可以用crf直接计算loss，也可以viterbi直接解码。

    4. Emit score映射层：将词语级的feature映射为label的emit score。比如BiLSTM的hidden size为768，label的数量为10，那么这一层就是一个768->10的全连接层。

    5. CRF：用于解决标签之间的依赖关系。计算loss时，不像softmax那样直接计算每个label的概率，而是计算整个序列给出emit score的条件概率。解码时，用viterbi算法输出一条条件概率最大的label序列。

    Bert-like model，BiLSTM，映射层都是直接调用库的，下面主要介绍字符到词语转换和CRF。

2. 字符到词语转换(`_char_feat_to_word_feat`)

    1. 输入：
        `char_feats`(char_seq_len x batch_size x hidden_size): 要转换的字符集特征；
        `word_id`(batch_size x char_seq_len)：这是Huggingface的fast tokenizer的输出之一。其将subword在句子中的位置映射到分词前的token在句子中的位置，由于special token本来不出现在句子里，所以映射到None。由于None不能转为tensor，预处理时将None转为-1；
        attention_mask(batch_size x char_seq_len)：将<PAD>映射为0，其他映射为1；

    2. 输出：
        `word_feats`(word_seq_len x batch_size x hidden_size)：由于CRF根据时间维度进行动态规划和计算句子分数等，所以这里将batch_size和seq_len的维度调换。
        `masks`(word_seq_len x batch_size)：标记哪些位置是真实的词语，哪些是padding的。

    3. 思路：
        由于这个层在Bert-like model后，所以需要考虑Bert-like model的特殊字符。由于特殊字符不对应任何label，所以直接去掉它们的特征，实现时使用切片索引获取非特殊字符的特征，这一步需要借助`attention_mask`的信息。然后根据`word_id`累和subwords得到word的特征，使用了`Tensor._index_add`方法；最后取平均，取平均需要先获得每个词语由多少subword组成，这里使用了`torch.bincount`。
    
    4. 代码
        具体实现见如下代码，其中包括了一些例子和更详细的解释：

    ```python
    def _char_feat_to_word_feat(self, char_feats, word_ids, attention_mask):
        # word_feats is the average of char_feats of the same word. The same word is defined by the same word_id.
        # char_feats: L x B x D
        # word_ids: B x L. Looks like: [[-1(CLS), 0, 1, 1, 2, -1(SEP), -1(PAD)], [-1, 0, 1, 1, 2, 2, -1], ...]
        # Output: word_feats: W x B x D, masks: W x B; W is the max word number in a batch; pad word_feats and masks with 0
    
        L, B, D = char_feats.shape
        # word_nums marks how many words each batch has
        word_nums = torch.max(word_ids, dim=1)[0] + 1 # B
        # char_nums marks how many non-special tokens(subwords) each batch has
        char_nums = torch.sum(attention_mask, dim=1) - 2 # B
        max_word_num = torch.max(word_nums)
        word_char_num = torch.zeros(max_word_num, B, dtype=torch.long, device=char_feats.device)

        # Count how many subwords every word has in each batch.
        for b in range(B):
            # e.g. [-1, 0, 0, 0, 1, 1, -1, -1, -1, -1] -> [3, 2, 0, 0, 0, 0]
            # Bincount can't handle negative values, so add 1 to word_ids
            word_char_num[:, b] = torch.bincount(word_ids[b] + 1, minlength=max_word_num+1)[1:]
        
        # Sum subword features.
        word_feats = torch.zeros(max_word_num, B, D, dtype=torch.float, device=char_feats.device)
        for b in range(B):
            # e.g. word_ids[0] = [-1, 0, 1, 1, 2, -1, -1], then word_feats[1] is the average of char_feats[2] and char_feats[3]
            # The symantic of the following line: for each valid char_feat(not special token) in char_feats,
            #   we add it to a word_feat in word_feats. The word_feat is determined by the 2nd parameter of index_add_, which is
            #   the valid word_id(no special token) of this batch.
            word_feats[:, b, :].index_add_(0, word_ids[b, 1:1+char_nums[b]], char_feats[1:1+char_nums[b], b, :])

        # Take means of sums.
        for b in range(B):
            word_num = word_nums[b]
            word_feats[:word_num, b, :] = word_feats[:word_num, b, :] \
                                        / word_char_num[:word_num, b].unsqueeze(1) # broadcast the hidden dimension

        # Get word-level masks.
        masks = torch.zeros(max_word_num, B, dtype=torch.bool, device=char_feats.device)
        for b in range(B):
            masks[:word_nums[b], b] = 1.
            # the rest of the mask are paddings, and they are already 0s in the initialization

        return word_feats, masks
    ```

3. CRF

CRF是一种无向图模型，并且它视input(emit score)为已知，只对hidden states进行建模(即label)。它一般有两个用途: 1. 计算某一个label序列的概率，涉及到计算这个序列的score和partition score；2. 使用viterbi算法获得概率最大（也就是score最大）的序列。
CRF建模的是label之间的依赖关系，所以它的score是由两部分组成的：1. emit score，即不考虑label之间的依赖关系的情况下，词语赋予某个label的的得分；2. transition score，即label之间的转移得分。特别地，我们还可以约束每个label作为起始和终止的得分。CRF的score计算公式如下：

$$
score(y) = \sum_{i=1}^{n} emit\_score_i(y_i) + \sum_{i=1}^{n-1} transition\_score(y_i, y_{i+1}) + start\_score(y_1) + end\_score(y_n)
$$
而partition score是对所有可能的label序列的score求和，即：

$$
partition\_score = \sum_{y \in Y} score(y)
$$
于是，label序列的概率就是：

$$
p(y|x) = \frac{score(y)}{partition\_score}
$$

实际中，emit score就是projection layer的结果，transition score用一个矩阵表示，这个矩阵要包括start和end的得分。

3.1 计算某一个label序列的得分(`CRF._cal_sent_score`)

1. 输入：
    `emit_scores`(seq_len x batch_size x label_num)：每个词语的emit score，代表了不考虑label之间的依赖关系的情况下，词语赋予某个label的得分；
    `labels`(seq_len x batch_size)：每个词语的label；
    `masks`(seq_len x batch_size)：标记哪些位置是真实的词语，哪些是padding的。

2. 输出：
    `score`(batch_size)：每个句子的score。

3. 思路：
    按照上述score的公式迭代计算即可。要注意不去加转移到padding的label的得分，也不加padding内部的转移得分，也不加padding的emit score，这可以用masks来实现。另外，因为padding的存在，结束得分放在最后处理。

4. 代码
    具体实现见如下代码：

```python

    def _cal_sent_score(self, emit_scores, labels, masks):
        L, B = masks.shape
        assert emit_scores.shape == (L, B, self.K)
        assert labels.shape == (L, B)
        score = torch.zeros(B, device=emit_scores.device) # B
        labels = torch.cat([torch.full((1, B), self.label_to_idx[START_LABEL], dtype=torch.long, device=emit_scores.device),
                            labels], dim=0) # Add a [Start] tag; (L + 1) x B
        for t, feat in enumerate(emit_scores):
            score += (self.transitions[labels[t + 1], labels[t]] + feat[range(B), labels[t + 1]]) * masks[t] # (B + B) x B -> B
        # Add the last transition to STOP_LABEL
        score += self.transitions[self.label_to_idx[STOP_LABEL], labels[torch.sum(masks, dim=0).long(), range(B)]]
        return score
```

3.2 计算所有可能label序列的得分总和(`CRF._cal_partition_score`)

1. 输入：
    `emit_scores`(seq_len x batch_size x label_num)：每个词语的emit score，代表了不考虑label之间的依赖关系的情况下，词语赋予某个label的得分；
    `masks`(seq_len x batch_size)：标记哪些位置是真实的词语，哪些是padding的。

2. 输出：
    `partition_score`(batch_size)：每个句子的partition score。

3. 思路：
    如果不用动态规划，那么有$K^T$种可能的label序列，其中K为标签数量，T为序列长度。这是无法有效计算的。但是可以利用动态规划的思想：
    我们将原问题："计算所有label序列的得分总和"，先变为"计算时间T时为label k的所有序列的得分总和"，后者的结果用logsumexp(此函数将代表一系列不相交事件的对数概率，也就是分数先求指数变为一系列概率，再求这些不相交事件的概率求和，最后取对数返回对数空间)可以转化为前者的结果。而"计算时间T时为label k的所有序列的得分总和"可以用动态规划解决。我们定义一个矩阵`alpha`，其中`alpha[t, k]`表示时间t时为label k的所有序列的得分总和。当子问题`alpha[t-1, k']`对于任何$k'$已经解决时，我们可以用如下公式解决子问题`alpha[t, k]`：

$$
\alpha[0, k] = start\_score[k] \\
\alpha[t, k] = Logsumexp_{k'}(\alpha[t-1, k'] + emit\_score_t[k] + transition\_score[k', k]) \   (t > 0, t < T) \\
\alpha[T, k] = Logsumexp_{k'}(\alpha[T-1, k'] + end\_score[k'])
$$
partition score如下：
$$
partition\_score = Logsumexp_k(\alpha[T, k])
$$

这样只需要$KT$个循环，就可以得到partition score。循环内部的计算是张量运算，所以效率很高。
同样地，我们需要注意padding的存在。在这里，我们要保证padding的时间步t保持上一个时间步的`alpha[t-1, k']`不变（对所有$k'$）。且结束得分要放在最后处理。

4. 代码
    具体实现见如下代码，其中变量名和代码都和上述公式对应：

    ```python
    def _cal_partition(self, emit_scores, masks):
        L, B = masks.shape
        # \alpha[0, k] = start\_score[k] \\
        forward_var_0 = torch.full((B, self.K), -10000., device=emit_scores.device)
        forward_var_0[:, self.label_to_idx[START_LABEL]] = 0.
        forward_var_t = forward_var_0
        # DP loop
        for t, feat in enumerate(emit_scores):
            forward_var_t_k = []  # The working forward variable at this timestep; B x K
            # Populate the next forward_var iteratively in K steps
            for k in range(self.K): # \alpha[t, k] = Logsumexp_{k'}(\alpha[t-1, k'] + emit\_score_t[k] + transition\_score[k', k])
                emit_score = feat[range(B), k].view(-1, 1).expand(B, self.K) # B -> B x K
                trans_score = self.transitions[k].view(1, -1).expand(B, self.K) # K -> B x K
                next_label_var = forward_var_t + (emit_score + trans_score) * masks[t].view(-1, 1) # B x K
                # Just keep the previous forward_var if the mask is 0; else sum the path scores
                next_label_var = torch.where(masks[t].bool(), log_sum_exp(next_label_var), forward_var_t[:, k]) # B
                forward_var_t_k.append(next_label_var)
            forward_var_t = torch.stack(forward_var_t_k, dim=1) # B x K
        forward_var_T = forward_var_t + self.transitions[self.label_to_idx[STOP_LABEL]]
        partition = log_sum_exp(forward_var_T)
        return partition
    ```

3.3 计算loss(`BiLSTMCRF.calculate_loss`(已废弃))

只需要将上述两个函数的结果相减即可。注意，由于我们的目标是最大化概率，所以我们要最大化score，也就是最小化负的score。所以我们的loss是负的score。还有，要对batch_size取平均。

代码如下：

```python
def calculate_loss(self, input_ids, attention_mask, word_ids, labels):
        labels = labels.T
        emit_score, word_masks = self._bilstm(input_ids, attention_mask, word_ids)
        score = self._cal_sent_score(emit_score, labels, word_masks)
        partition = self._cal_partition(emit_score, word_masks)
        loss = partition - score
        return loss.mean(dim=1)
```

3.4 解码(`CRF.predict`)

解码的目标是找到概率最大的label序列。这个问题和计算partition score的问题是一样的，只是我们要找到最大的score对应的序列，而不是求和。我们可以用viterbi算法解决这个问题。viterbi算法和前述的动态规划结构完全一致，只是把logsumexp换为max，然后需要记录每个子问题的解序列。我们定义一个矩阵`alpha`，其中`alpha[t, k]`表示时间t时为label k的所有序列的最大得分。当子问题`alpha[t-1, k']`对于任何$k'$已经解决时，我们可以用如下公式解决子问题`alpha[t, k]`：

$$
\alpha[0, k] = start\_score[k] \\
\alpha[t, k] = Max_{k'}(\alpha[t-1, k'] + emit\_score_t[k] + transition\_score[k', k]) = Max_{k'}(\alpha[t-1, k'] + transition\_score[k', k]) + emit\_score_t[k] \   (t > 0, t < T) \\
\alpha[T, k] = Max_{k'}(\alpha[T-1, k'] + end\_score[k'])
$$

由于我们并不是想要最大的score，而是得到最大score对应的label序列，所以我们还需要记录每个子问题的解序列。我们定义一个矩阵`bptrs`(back pointers)，其中`bptrs[t, k]`表示得到时间t时为label k中的所有序列的最大得分的序列的上一个时间步的label。这个矩阵的构建和`alpha`矩阵一样，只是把max换为argmax。最后，我们从最后一个时间步([STOP])开始，根据`bptrs`逐步回溯，然后去掉[START]就可以得到最大score对应的label序列。


