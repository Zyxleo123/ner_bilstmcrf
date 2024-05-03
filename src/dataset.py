from torch.utils.data import Dataset
from .variable import *

class NERDataset(Dataset):
    def __init__(self, split, tag_to_ix):
        self.tag_to_ix = tag_to_ix
        if split in ['train', 'dev', 'toy', 'train_']:
            self.text, self.labels = self.load_annotated_data(split)
        elif split == 'test':
            self.text, self.id = self.load_test_data(split)
        else:
            raise ValueError('split must be train, dev or test')
        
    def load_annotated_data(self, split):
        if split not in ['train', 'dev', 'toy', 'train_']:
            raise ValueError('split must be train or dev or toy')
        words = []
        labels = []
        with open('data/' + split + '.csv', 'r') as f:
            for i, line in enumerate(f):
                # skip 1st and empty lines
                line = line.strip()
                if line == '' or i == 0:
                    continue
                idx = line.rfind(',')
                word = line[:idx]
                label = line[idx+1:]
                words.append(word)
                labels.append(label)
                
        # split on '。' and '.' and '!' and '?' and '？' and '！'
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

        # in case last sentence does not end with a splitter
        if start < len(words):
            sentence = words[start:]
            sentence_label = labels[start:]
            sentences.append(sentence)
            sentence_labels.append(sentence_label)

        assert len(sentences) == len(sentence_labels)
        return sentences, sentence_labels

    def load_test_data(self, split):
        if split != 'test':
            raise ValueError('split must be test')
        ids = []
        words = []
        with open('data/' + split + '.csv', 'r') as f:
            for i, line in enumerate(f):
                # skip 1st and empty lines
                if line.strip() == '' or i == 0:
                    continue
                idx = line.find(',')
                id = int(line[:idx])
                word = line[idx+1:]
                ids.append(id)
                words.append(word)
        
        # split on '。' and '.' and '!' and '?' and '？' and '！'
        splitter = ['。', '.', '!', '！', '?', '？']
        sentences = []
        sentence_ids = []
        start = 0
        for i in range(len(words)):
            if words[i] in splitter:
                sentence = words[start:i+1]
                sentence_ids = ids[start:i+1]
                start = i+1
                sentences.append(sentence)
                sentence_ids.append(sentence_ids)

        # in case last sentence does not end with a splitter
        if start < len(words):
            sentence = words[start:]
            sentence_ids = ids[start:]
            sentences.append(sentence)
            sentence_ids.append(sentence_ids)

        assert len(sentences) == len(sentence_ids)
        return sentences, sentence_ids
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        if self.split in ['train', 'dev']:
            return self.text[index], self.labels[index]
        elif self.split == 'test':
            return self.text[index], self.id[index]
