import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorWithPadding
from .variable import PAD_LABEL, LABEL_TO_IDX

class NERDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # need to select these because tokenizer.pad does not pad other names even if they are in the features
        # then it still transfer all keys to tensors, which's impossible because of some of them have different lengths
        need_padding = ['input_ids', 'attention_mask'] 
        batch = self.tokenizer.pad([{k: feature[k] for k in need_padding} for feature in features], padding='longest', return_tensors='pt')
        max_length = batch['input_ids'].shape[1]

        # for word_ids, it needs to be padded to the max length that equals to the max length of tokens.
        # word_ids = [feature['word_ids'] for feature in features]
        # word_ids = [list(map(lambda x: -1 if x is None else x, word_id)) for word_id in word_ids]
        # word_ids = [word_id + [-1] * (max_length-len(word_id)) for word_id in word_ids]
        # word_ids = torch.tensor(word_ids)
        word_ids = [torch.tensor(feature['word_ids']) for feature in features]
        # word_ids = [torch.tensor(list(map(lambda x: -1 if x is None else x, word_id))) for word_id in word_ids]
        word_ids = pad_sequence(word_ids, padding_value=-1, batch_first=True)

        if 'labels' not in features[0]:
            return {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'word_ids': word_ids}

        labels = [torch.tensor(feature['labels']) for feature in features]
        # labels = [torch.tensor([LABEL_TO_IDX[label] for label in batch_label]) for batch_label in labels]
        labels = pad_sequence(labels, padding_value=0, batch_first=True)
        return {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'], 'word_ids': word_ids, 'labels': labels}
