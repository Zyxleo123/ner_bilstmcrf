# PJ2: 基于LSTM-CRF的序列标注

> 赵一溪 21307130052

## 1. 代码

### 1.1 模型

#### 总架构

> Bert-like model(as embedding) -> BiLSTM -> 字符到词语转换 -> Emit score映射层(BiLSTM hidden size -> #Label) -> CRF

1. **Bert-like model：**
    
    载入Huggingface的预训练模型，用于生成上下文有关的高质量词向量

2. **BiLSTM：**

    和后面的所有层一样为fine tune的预测头。用于提取适合本次数据集的上下文有关的特征。之所以不去掉这一层，是因为否则需要把task-specific的特征提取放到Bert-like model中，但是bert-like model的参数量太大相较数据量，不适合fine-tuning。

3. **字符到词语转换：**

    用于将BiLSTM的输出映射到每个词语的输出。这一层将[<CLS>, subword_embedding_1_1, subword_embedding_1_2, subword_embedding_2_1, ..., subword_embedding_n_3, <SEP>, <PAD>, <PAD>, ..., <PAD>]映射到[word_embedding_1, word_embedding_2, ..., word_embedding_n]。目的是使得label和feature的长度对齐，这样才可以用crf直接计算loss，也可以viterbi直接解码。

4. **Emit score映射层：**

    将词语级的feature映射为label的emit score。比如BiLSTM的hidden size为768，label的数量为10，那么这一层就是一个768->10的全连接层。

5. **CRF：**

    用于解决标签之间的依赖关系。计算loss时，不像softmax那样直接计算每个label的概率，而是计算整个序列给出emit score的条件概率。解码时，用viterbi算法输出一条条件概率最大的label序列。

Bert-like model，BiLSTM，映射层都是直接调用库的，下面主要介绍字符到词语转换和CRF。

#### 字符到词语转换(`_char_feat_to_word_feat`)

1. **输入：**

    `char_feats`(char_seq_len x batch_size x hidden_size): 要转换的字符集特征；
    `word_id`(batch_size x char_seq_len)：这是Huggingface的fast tokenizer的输出之一。其将subword在句子中的位置映射到分词前的token在句子中的位置，由于special token本来不出现在句子里，所以映射到None。由于None不能转为tensor，预处理时将None转为-1；
    `attention_mask`(batch_size x char_seq_len)：将<PAD>映射为0，其他映射为1；

2. **输出：**

    `word_feats`(word_seq_len x batch_size x hidden_size)：由于CRF根据时间维度进行动态规划和计算句子分数等，所以这里将batch_size和seq_len的维度调换。
    `masks`(word_seq_len x batch_size)：由于不同batch含有词语个数不同，用masks标记哪些位置是真实的词语，哪些是padding的。

3. **思路：**

    由于这个层在Bert-like model后，所以需要考虑Bert-like model的特殊字符。由于特殊字符不对应任何label，所以直接去掉它们的特征，实现时使用切片索引获取非特殊字符的特征，这一步需要借助`attention_mask`的信息。然后根据`word_id`累和subwords得到word的特征，使用了`Tensor._index_add`方法；最后取平均，取平均需要先获得每个词语由多少subword组成，这里使用了`torch.bincount`。
    
4. **代码**

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

    和Bert-like model, projection放在一起，获得CRF的emit scores和word_masks的代码如下：

    ```python
    def _bilstm(self, input_ids, attention_mask, word_ids):
        embeds = self.bert_like_model(input_ids, attention_mask=attention_mask).last_hidden_state
        embeds = embeds.permute(1, 0, 2) # change to time-step first
        feats, self.state = self.lstm(embeds)
        feats, word_masks = self._char_feat_to_word_feat(feats, word_ids, attention_mask)
        emit_scores = self.projection(feats)
        return emit_scores, word_masks
    ```

#### CRF


- CRF是一种无向图模型，并且它视input(emit score)为已知，只对hidden states进行建模(即label)。它一般有两个用途: 1. 计算某一个label序列的概率，涉及到计算这个序列的score和partition score；2. 使用viterbi算法获得概率最大（也就是score最大）的序列。
CRF建模的是label之间的依赖关系，所以它的score是由两部分组成的：1. emit score，即不考虑label之间的依赖关系的情况下，词语赋予某个label的的得分；2. transition score，即label之间的转移得分。特别地，我们还可以约束每个label作为起始和终止的得分。CRF的score计算公式如下：
由于本任务中，只存在类似"B必须转移到相同类型的M或E"的相邻label的约束，所以只用transition matrix建模label间约束是足够的。

$$
score(y) = \sum_{i=1}^{n} emit\_score_i(y_i) + \sum_{i=1}^{n-1} transition\_score(y_i, y_{i+1}) + start\_score(y_1) + end\_score(y_n)
$$

- 而partition score是对所有可能的label序列的score求和，即：

$$
partition\_score = \sum_{y \in Y} score(y)
$$

- 于是，label序列的概率就是：

$$
p(y|x) = \frac{score(y)}{partition\_score}
$$

- 实际中，emit score就是projection layer的结果，transition score用一个矩阵表示，这个矩阵要包括start和end的得分。

- 计算某一个label序列的得分(`CRF._cal_sent_score`)

1. **输入：**


    `emit_scores`(seq_len x batch_size x label_num)：每个词语的emit score，代表了不考虑label之间的依赖关系的情况下，词语赋予某个label的得分；
    `labels`(seq_len x batch_size)：每个词语的label；
    `masks`(seq_len x batch_size)：标记哪些位置是真实的词语，哪些是padding的。

2. **输出：**


    `score`(batch_size)：每个句子的score。

3. **思路：**


    按照上述score的公式迭代计算即可。要注意不去加转移到padding的label的得分，也不加padding内部的转移得分，也不加padding的emit score，这可以用masks来实现。另外，因为padding的存在，结束得分放在最后处理。

4. **代码**


    具体实现见如下代码：

```python

    def _cal_sent_score(self, emit_scores, labels, masks):
        L, B = masks.shape
        score = torch.zeros(B, device=emit_scores.device) # B
        labels = torch.cat([torch.full((1, B), self.label_to_idx[START_LABEL], dtype=torch.long, device=emit_scores.device),
                            labels], dim=0) # Add a [Start] tag; (L + 1) x B
        for t, feat in enumerate(emit_scores):
            score += (self.transitions[labels[t + 1], labels[t]] + feat[range(B), labels[t + 1]]) * masks[t] # (B + B) x B -> B
        # Add the last transition to STOP_LABEL
        score += self.transitions[self.label_to_idx[STOP_LABEL], labels[torch.sum(masks, dim=0).long(), range(B)]]
        return score
```

- 计算所有可能label序列的得分总和(`CRF._cal_partition_score`)

1. **输入：**


    `emit_scores`(seq_len x batch_size x label_num)：每个词语的emit score，代表了不考虑label之间的依赖关系的情况下，词语赋予某个label的得分；
    `masks`(seq_len x batch_size)：标记哪些位置是真实的词语，哪些是padding的。

2. **输出：**


    `partition_score`(batch_size)：每个句子的partition score。

3. **思路：**


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

4. **代码**


    具体实现见如下代码。实现时，并不会真正存储一整个矩阵的子问题解，毕竟我们只需要上一个时间步的解。于是只需要一个向量即可（见`forward_var_t`）。剩余部分都和上述公式一致。

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

    将CRF放在整个模型当中，计算loss时，需要计算score和partition score，然后只需要将上述两个函数的结果相减即可。由于的目标是最大化概率，所以我们要最大化score，也就是最小化负的score。所以loss是负的score。还有，要对batch_size取平均。

    ```python
    def calculate_loss(self, input_ids, attention_mask, word_ids, labels):
        labels = labels.T
        emit_score, word_masks = self._bilstm(input_ids, attention_mask, word_ids)
        score = self._cal_sent_score(emit_score, labels, word_masks)
        partition = self._cal_partition(emit_score, word_masks)
        loss = partition - score
        return loss.mean(dim=1)
    ```

- 解码(`CRF.predict`)

1. **输入：**


    `emit_scores`(seq_len x batch_size x label_num)：每个词语的emit score，代表了不考虑label之间的依赖关系的情况下，词语赋予某个label的得分；
    `masks`(seq_len x batch_size)：标记哪些位置是真实的词语，哪些是padding的。

2. **输出：**


    `best_paths`(seq_len x batch_size)：每个batch的最优序列。
    `best_scores`(batch_size)：每个batch的最优序列的得分。

3. **思路**


    解码的目标是找到概率最大的label序列。这个问题和计算partition score的问题是一样的，只是我们要找到最大的score对应的序列，而不是求和。我们可以用viterbi算法解决这个问题。viterbi算法和前述的动态规划结构完全一致，只是把logsumexp换为max，然后需要记录每个子问题的解序列。我们定义一个矩阵`alpha`，其中`alpha[t, k]`表示时间t时为label k的所有序列的最大得分。当子问题`alpha[t-1, k']`对于任何$k'$已经解决时，我们可以用如下公式解决子问题`alpha[t, k]`：

    $$
    \alpha[0, k] = start\_score[k] \\
    \alpha[t, k] = Max_{k'}(\alpha[t-1, k'] + emit\_score_t[k] + transition\_score[k', k]) = Max_{k'}(\alpha[t-1, k'] + transition\_score[k', k]) + emit\_score_t[k] \   (t > 0, t < T) \\
    \alpha[T, k] = Max_{k'}(\alpha[T-1, k'] + end\_score[k'])
    $$

    至此`best_scores`可以计算。

    由于我们并不是想要最大的score，而是得到最大score对应的label序列，所以我们还需要记录每个子问题的解序列。我们定义一个矩阵`bptrs`(back pointers)，其中`bptrs[t, k]`表示得到时间t时为label k中的所有序列的最大得分的序列的上一个时间步的label。这个矩阵的构建和`alpha`矩阵一样，只是把max换为argmax。最后，我们从最后一个时间步([STOP])开始，根据`bptrs`逐步回溯，然后去掉[START]就可以得到最大score对应的label序列。

4. **代码**


    具体实现见如下代码。实现时同样只需要一个向量即可（见`forward_var_t`）；且不用max函数，而是先用argmax得到每个batch的最好前一时间步label的**索引**，然后用这个索引去取得分，这样间接完成max操作。

    ```python
    def _predict(self, emit_scores, masks):
        L, B = masks.shape
        bptrs = []
        # \alpha[0, k] = start\_score[k] \\
        forward_var_0 = torch.full((B, self.K), -10000., device=emit_scores.device)
        forward_var_0[:, self.label_to_idx[START_LABEL]] = 0. # B x K
        forward_var_t = forward_var_0

        # DP loop
        for t, emit_score in enumerate(emit_scores):
            bptrs_t = []
            forward_var_t_k = []
            # Populate the next forward_var iteratively in K steps
            for next_label in range(self.K):
                # Don't add emission score here since it's the same for all paths and doesn't affect the argmax
                next_label_var = forward_var_t + self.transitions[next_label] # B x K
                best_label_ids = torch.argmax(next_label_var, dim=1) # B
                next_label_var = torch.where(masks[t].bool(), next_label_var[range(B), best_label_ids], forward_var_t[:, next_label]) # B
                bptrs_t.append(best_label_ids)
                forward_var_t_k.append(next_label_var)
            # Now add in the emission scores, and assign forward_var to the set of viterbi variables we just computed
            forward_var_t = torch.stack(forward_var_t_k, dim=1) + emit_score * masks[t].view(-1, 1) # B x K
            bptrs.append(torch.stack(bptrs_t, dim=1)) # B x K

        bptrs = torch.stack(bptrs, dim=0) # L x B x K
        # Transition to STOP_LABEL
        forward_var_T = forward_var_t + self.transitions[self.label_to_idx[STOP_LABEL]] # B x K
        best_label_ids = torch.argmax(forward_var_T, dim=1) # B
        best_scores = forward_var_T[range(B), best_label_ids]
        
        best_paths = []
        for b in range(B):
            # Follow the back pointers to decode the best path.
            best_label_id = best_label_ids[b]
            best_path = [best_label_id]
            real_L = torch.sum(masks[:, b]).long()
            for bptrs_t in reversed(bptrs[:real_L, b]):
                best_label_id = bptrs_t[best_label_id]
                best_path.append(best_label_id)
            # Pop off the start label (we dont want to return that to the caller)
            start = best_path.pop()
            best_path.reverse()
            best_paths.append(best_path)
        return best_scores, best_paths
    ```
    
    将CRF放在整个模型当中，解码时，只需要调用上述函数即可。

    ```python
    def predict(self, input_ids, attention_mask, word_ids):
        emit_scores, word_masks = self._bilstm(input_ids, attention_mask, word_ids)
        pred = self.crf.decode(emit_scores, mask=word_masks)
        return pred
    ```



### 1.2 数据预处理

需要将原始数据处理成Bert-like model的输入。

#### 数据格式转化

原始数据是词语级别的，但是我们希望模型接受一个句子的输入，这样可以用上下文帮助分类。于是需要把一连串不相交的词语group成一个句子，实验中使用表示句子终止的标点符号作为句子的分隔符：也就是如果看到了一个词语等于`。`，`！`，`？`等符号，就截断当前句子，开始新的句子。但是实际当中，按照这样分句会使得句子分词以后长度大于512，也就是Bert-like model的输入长度限制。于是进一步地，如果一句句子到了128个词语尚未结束，那么就通过逗号等符号分割句子。经过测试，128是一个即保证句子不会太短，又保证句子不会溢出的合适的长度。

由于篇幅原因，此处只展示分句部分的代码：

```python
# split on these symbols
splitter = ['。', '.', '．', '!', '！', '?', '？']
sentences = []
sentence_labels = []
start = 0
for i in range(len(words)):
    if words[i] in splitter:
        sentence = words[start:i+1]
        sentence_label = labels[start:i+1]
        start = i+1
        sentences.append(sentence)     
        sentence_labels.append(sentence_label)
    # in case sentence is too long
    if i - start > 128 and words[i] in ['，', ',', '、', '、']:
        sentence = words[start:i+1]
        sentence_label = labels[start:i+1]
        start = i+1
        sentences.append(sentence)     
        sentence_labels.append(sentence_label)
```

使用pytorch的Dataset包装转化后的数据，其中某一个样本的数据格式如下：

```python
{
    'text': ['我', '住在', '北京', '市', '海淀', '区', '中关村', '街道', '上', '。'],
    'label': ['O', 'O', 'B-GPE', 'E-GPE', 'B-GPE', 'E-GPE', 'B-LOC', 'E-LOC', 'O', 'O']
}
```

#### 数据编码

接下来，需要把包括text和label的字符串都转化为数字，顺便得到没有pad的attention_mask和word_ids。

对于label来说，只要随便给不同的自然数，并且保证解码时使用相同的映射即可。

```python
# train.py line 47
encoding['labels'] = [[LABEL_TO_IDX[y] for y in b] for b in example['labels']]
```

对于text，需要按照Bert-like model的tokenizer vocabulary进行转化；并且由于Bert-like model使用subword预训练，还要先将词语分割为subword tokens，再加上模型特殊的special tokens，最后按照vocabulary转为input_ids。这一步只需要调用Huggingface的tokenizer即可。tokenizer输出的batch encoding不仅包括input_ids和attention_mask，还包括word_ids，其中前两者已经都为数字，而word_ids中special tokens对应的还是None，需要转为-1。

```python
# train.py line 44~46
encoding = tokenizer(example["text"], is_split_into_words=True)
encoding['word_ids'] = [encoding.word_ids(b) for b in range(len(example['labels']))]
encoding['word_ids'] = [list(map(lambda x: -1 if x is None else x, word_id)) for word_id in encoding['word_ids']]
```

总体来说，从Dataset转为用数字编码的数据的代码如下：

```python
def tokenize(example):
    encoding = tokenizer(example["text"], is_split_into_words=True)
    encoding['word_ids'] = [encoding.word_ids(b) for b in range(len(example['labels']))]
    encoding['word_ids'] = [list(map(lambda x: -1 if x is None else x, word_id)) for word_id in encoding['word_ids']]
    encoding['labels'] = [[LABEL_TO_IDX[y] for y in b] for b in example['labels']]
    return encoding
```

例：

![mapped](./images/mapped.png)

#### 批次collate

生成批次时，需要考虑到不同批次拥有不同的tokens数量。需要把每个输入的input_ids, attention_mask, labels, word_ids都pad到最长的长度。
    
其中input_ids和attention_mask的padding由模型的tokenizer pad方法负责，这样它能按照[PAD]的id来正确地pad input_ids。
    
而此方法不能处理labels和word_ids，所以使用更低层的方法pad这两个序列：torch.nn.utils.rnn.pad_sequence，它接受一个list，每个元素是一个LxD的tensor，L可变，D不可变。函数stack这些tensor，并且pad给定的值到最长的长度。

代码如下：

```python
class NERDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # need to select these because tokenizer.pad does not pad other names even if they are in the features
        # then it still transfer all keys to tensors, which's impossible because of some of them have different lengths
        need_padding = ['input_ids', 'attention_mask'] 
        batch = self.tokenizer.pad([{k: feature[k] for k in need_padding} for feature in features], padding='longest', return_tensors='pt')
        word_ids = [torch.tensor(feature['word_ids']) for feature in features]
        word_ids = pad_sequence(word_ids, padding_value=-1, batch_first=True)
        if 'labels' not in features[0]:
            return {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'word_ids': word_ids}
        labels = [torch.tensor(feature['labels']) for feature in features]
        labels = pad_sequence(labels, padding_value=0, batch_first=True)
        return {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'word_ids': word_ids, 'labels': labels}
```

例:

![collated](./images/collated.png)


实验的训练，验证和测试等流程都交给了Pytorch Lightning的Trainer来处理，只需要写出基本的train_step, val_step, on_validation_epoch_end等方法即可。

### 1.3 训练

首先需要初始化模型（于Lightning Module的`__init__`方法中），optimizer&scheduler（于Lightning Module的`configure_optimizers`方法中），数据加载器（`Trainer.fit`的传入参数中）。

然后只需要写出一个返回batch loss的train_step， `Trainer.fit`就可以正确训练模型。训练模式的开启，loss的backward，optimizer的step等都由Trainer来处理。

    ```python
    def training_step(self, batch, batch_idx):
        loss = self.model.calculate_loss(**batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    ```

### 1.4 验证

这里需要先写validation step的方法，其调用前述的calculate_loss和predict，得到的validation step loss和validation step prediction累积到2个列表里，以便在validation epoch结束时计算平均loss和总的Macro F1。为了方便观察，最好把被编码的标签转回字符串。

    ```python
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, word_ids, gt = batch['input_ids'], batch['attention_mask'], batch['word_ids'], batch['labels']
        pred = self.model.predict(input_ids=input_ids, attention_mask=attention_mask, word_ids=word_ids)
        loss = self.model.calculate_loss(input_ids=input_ids, attention_mask=attention_mask, word_ids=word_ids, labels=gt)
        pred = [[IDX_TO_LABEL[int(y)] for y in b] for b in pred]
        gt = [[IDX_TO_LABEL[int(y)] for y in b] for b in gt]
        self.val_gts.extend(gt)
        self.val_preds.extend(pred)
        self.val_losses.append(loss.item())
        return gt, pred
    ```

最后在validation epoch结束时计算平均loss和总的Macro F1。平均loss的计算直接除validation step number即可，这么做的正确性由每一个step有相同数量的batch保证(除了最后一个step)。实验中重新实现了`classification_report`函数来计算Macro F1：先对每一句句子统计每一个label的TP, FP, FN，然后计算每一个label的Accuracy, Precision, Recall, F1, 最后对所有label的F1取平均得到Macro F1，返回的字典中包括了label-wise的F1和Macro Accuracy, Precision, Recall, F1。`classification_report`的代码冗长，此处不展示。on_validation_epoch_end的代码如下:

    ```python
    def on_validation_epoch_end(self):
        report = classification_report(self.val_gts, self.val_preds, LABELS)
        self.log_dict(report, prog_bar=True)
        self.log('val_f1', report['macro avg_f1-score'], prog_bar=True)
        avg_loss = sum(self.val_losses) / len(self.val_losses)
        self.log('val_loss', avg_loss, prog_bar=True)
        self.val_gts.clear()
        self.val_preds.clear()
        self.val_losses.clear()
    ```

### 1.5 测试

测试比较简单，因为只需要输出prediction，写到文件中即可。测试的单步和验证的单步基本一致，只是不需要计算loss。测试的代码如下：

```python
def test_step(self, batch, batch_idx):
    input_ids, attention_mask, word_ids = batch['input_ids'], batch['attention_mask'], batch['word_ids']
    pred = self.model.predict(input_ids=input_ids, attention_mask=attention_mask, word_ids=word_ids)
    self.test_preds.extend(pred)
    return pred

def on_test_epoch_end(self):
    self.test_preds = sum(self.test_preds, [])
    self.test_preds = [IDX_TO_LABEL[int(y)] for y in self.test_preds]
    run_name = get_run_name(self.hparams)
    with open(f'outputs/{run_name}.txt', 'w') as f:
        f.write('expected\n') # id is added afterwards
        for p in self.test_preds:
            f.write(f'{p}\n')
```

## 2. 实验过程和结果

### 2.1 BiLSTM的padding问题

首先，我注意到了一个问题：BiLSTM如何处理padding？不像单向的循环神经网络，BiLSTM的每一个时间步的输出取决于一整个序列，其中也包括padding。经过分析与验证，我发现需要3点才能保证padding不影响其它时间步的状态：1. padding的特征是全0；2. 初始状态是全0；3. BiLSTM不使用bias，也就是bias是全0。第一点可以通过乘attention mask实现；第二点通过输入BiLSTM初始状态None实现；第三点通过初始化BiLSTM`bias=False`实现。

验证如下，分别向bias=False的bilstm输入2个unbatched输入，两个输入的1st time step=[1,1]。第一个输入(pad_1_step)有一个全0 padding，第二个输入(pad_2_steps)有两个全0 padding。如果padding没有任何影响的话，那么两个输入的1st time step输出应该是一样的。实验结果如下：

![nobias](./images/nobias.png)

如果padding非0：

![nonzero_pad](./images/nonzero_pad.png)

如果状态非0：

![nonzero_state](./images/nonzero_state.png)

如果bias非0：

![nonzero_bias](./images/nonzero_bias.png)

实验中，我假设padding的影响是一定不好的，于是我按照全0的方法处理了padding。我当时实际上忽视了Bert会将全0的变为非全0的特征，我也错误地没有紧接用attention mask再把它们清零。于是我在全部实验完成后，又实验了正确实现padding的处理的结果，并且外加了一组实验，其中权重，初始状态，bias和padding都是非0的，相当于无视这个问题。实验结果如下：

可见，这个忧虑可能是多余的(至少当收敛本来就不成问题时)。我猜测也许是因为padding对状态进行的改变类似也相当于是为真正的输入作随机的状态初始化（得益于lstm的tanh函数限制了状态的范围）。所以padding的影响并不是很大。在unbatched的场景下（比如validation和test），初始化非0状态有另外的好处：可以使下一个句子用上一个句子的状态作为初始状态，其效果没有实验验证。

### 2.2 计算性能

#### _char_feat_to_word_feat

(此部分由于进行于实验前期，其log和代码都已经被删除，所以只能通过回忆来写)

一开始，我实现了一个直观简单的版本。使用2层循环，第一层循环遍历batch，第二层循环遍历每个时间步。这么做耗时很长，通过计时可知每一次要花费7s左右（batch_size大概是12），于是成为了训练更多epoch的瓶颈之一。

于是着手优化这个函数，优化的方式就是用张量运算代替循环。这样的好处是，张量运算是高度并行的，所以可以利用GPU的并行计算能力。我找到两个函数，`torch.bincount`和`torch.index_add_`，这样使得时间步的循环并行化了。最后，batch size=12的情况下，每次调用此函数需要0.08s，如下图`Convert time`所示。

![convert](./images/convert.png)

不过，当batch size上升，自然这个函数的耗时会线性增加，慢慢地又成为了瓶颈之一。由于最后每个epoch耗时可以接受(~10min)，没有进一步优化。

#### CRF

CRF在上文中的实现的线性规划是两层循环：外层为时间步，内层为label。可见，这个实现已经把batch给并行化了。由于动态规划限制了后面的时间步必须等待前面的时间步的结果，所以时间步不可以并行化。但是Label确可以并行化。我没有自己实现CRF的张量化版本，而是直接调用了pytorch-crf的实现，它的实现就是只用了一个时间步的循环的。原来CRF一次损失计算要花费3s左右，现在只需要0.38s，如上图`CRF time`所示。

### 2.3 sgd vs adamw

实验中发现adamw的效果远好于sgd。我猜测是因为adamw的自适应学习率可以更好地适应不同的参数，而模型中CRF和LSTM的参数的数量差异很大，CRF的transition matrix中每个参数的影响力远大于LSTM的任一个参数，所以adamw更适合这个任务。下面是当时的Tensorboard的结果，F1更高的那一组是adamw的结果。

![adamw](./images/adamw.png)

于是我在最后的实验中，都使用了adamw。

### 2.4 lr的观察

下面是实验中期，F1可以达到40~50%时，使用`lr=5e-3`, `lr=5e-4`, `lr=1e-3`训练的`train_loss`，自上到下分别用`lstm_state_dim=1024`, `lstm_state_dim=512`, `lstm_state_dim=256`。

![lr005_1](./images/lr005_1.png)

![lr005_2](./images/lr005_2.png)

![lr005_3](./images/lr005_3.png)

共同之处在于，`lr=5e-3`的训练都很快就收敛了，但是最后收敛效果不如其它lr；不同之处在于随着lstm_state_dim的减小，`lr=5e-3`最后的收敛效果和其它lr的差距越来越小。lr越大，就越容易跳过最优点，所以最后的收敛效果不如其它lr。而lstm_state_dim越小，越适合用大的lr，所以`lr=5e-3`在lstm_state_dim=256的情况下表现最好。

下面是实验后期，F1可以达到~90%时，使用`lr=5e-3`, `lr=5e-4`, `lr=1e-3`训练的`train_loss`，自上到下分别用`lstm_state_dim=1024`, `lstm_state_dim=512`, `lstm_state_dim=256`，全部没有使用scheduler。

![lr005_5](./images/lr005_5.png)

![lr005_6](./images/lr005_6.png)

![lr005_7](./images/lr005_7.png)

可以明显地发现lr越大，则train loss的震荡越大，收敛效果越差；而lstm_state_dim越小，train loss的震荡越小，收敛效果越好。而如果都使用合适的学习率，不同的lstm_state_dim可以取得相似的收敛效果(如图：`smoothed train loss=0.0019`)。

也可以观察到相似的震荡发生在validation F1上，但是这个最后都可以达到一个比较好的值。在此不展示截图。

值得一提的是，train loss在这个任务下和validation F1的关系并不直接，比如，`lstm_state_dim=1024`的`lr=5e-3`虽然最后收敛的最差，但是validation F1却是最高的。这应该是由于即便模型对某个预测的自信度下降，也不影响它预测正确的标签；同理，即便模型对某个预测的自信度上升，也不影响它预测错误的标签。

![lr005_4](./images/lr005_4.png)

由于validation F1作为最终的评价指标的直接反映，和其关于lr的随机性，后续实验都不假设哪个lr最好，而是都进行搜索。

### 2.5 Scheduler的选择

为了解决类似上述静态学习率导致的问题，我尝试了3种scheduler：`Linear`，`CosineAnnealWarmRestarts`，`OneCycleLR`。Linear直接从初始lr线性下降到0；CosineAnnealWarmRestarts周期性地从初始lr下降到0；OneCycleLR先从最小lr上升到最大lr，然后再下降到最小lr。以下是`lr=5e-3`的实验结果：

下图展示了`lstm_state_dim=512`，`lr=5e-3`的3种scheduler和无scheduler的`train_loss`(Onecycle的warm up为30%步；CosineAnnealWarmRestarts的周期为$\#epochs//5$不变；其余都用默认参数)：

![scheduler_loss](./images/scheduler_loss.png)

Onecycle的loss一开始减少的很慢，但是由于比较稳定，最后收敛的最好；Linear由于最后lr也减少到0，收敛的也很好，与Onecycle相似；Anneal由于在训练过程中相较其它scheduler最多次使用大的lr，于是收敛速度整体最快，但是由于不断地重启，所以最后收敛效果没有其它好（这也是由于Onecycle和Linear收敛的太好了的关系）；无scheduler的收敛效果最差，它在最后还使用最大的lr，导致不能较好收敛。

在validation的相关指标上，我并不能观察到scheduler的这些特征，这再次印证了training的收敛效果和validation的指标不一定直接有关。除了onecycle：validation loss低，且稳定；validation F1稳步上升，但是不高。在此不展示截图。

我最后选择了CosineAnnealWarmRestarts。因为我认为训练更多epochs的情况下，它相较于其它调度器可以发现更多的最优点，而可能某一个就能使得validation F1更高。

### 2.6 数据增强

实验中期时，试图增强数据来提高少数类的F1。由于我只是想要让模型多预测这些少数类，而这些预测的accuracy不重要，这样以期提高少数类的recall。这个过程必然会降低其它类的recall，于是最后需要取得一个折衷。可惜的是，我使用的数据增强似乎过于简单了：我将少数类的数据简单地复制了几次随机插入到原数据中，用这种方式reward模型多预测少数类。模型似乎并没有“理睬”这个改变，也就是仍然不怎么倾向于预测少数类。我猜测这是因为模型只是比起以前更加倾向于将**看过**的少数类更多的确实预测为该少数类，但是由于数据增强的方式过于简单，模型的行为不能泛化到其它数据。拿二维特征上的二分类打比方，这么增强相当于把决策边界的某一处极大地扭曲到某一个数据点周围把它半包住，但是其余处丝毫没有改变。我认为更好的增强应该涉及到更多的数据：这些数据可以概括少数类的特征，而且覆盖一定的空间，这样模型才能更好地泛化到其它数据。比如要把人名换成其它人名，地名换成其它地名，企业名换成其它企业名等等，然后标注它们所希望模型预测更多的label。但是由于时间有限，我没有实现这个想法。

### 2.7 预训练模型的影响

实验中影响F1最大的其实是预训练模型的选择。使用任务无关的预训练Bert只能达到40-50的F1，但是用在OntoNotes5上预训练的Bert可以达到90的F1。这是因为相当于使用的模型多用了

### 2.8 Ablation Study

分别将Bert去除（换成随机初始化的embedding），BiLSTM去除，以及交换BiLSTM和`_char_feat_to_word_feat`的位置，观察F1的变化。(CRF去除需要改变损失计算和预测的代码，所以没有实验)

使用的超参数：`bert_lr=0(freeze)`，`lstm_lr=5e-3`，`lstm_state_dim=256`，`scheduler=OneCycleLR（因为比较稳定，适合做实验）`，`adamw`，`epochs=10`，`batch_size=12`，`pretrained_model_name=ckiplab/bert-base-chinese-ner`。

| Model | F1 |
| --- | --- |
| noBert | 0.92 |
| noBiLSTM | 0.92 |
| swap | 0.92 |
| baseline | 0.8842 |