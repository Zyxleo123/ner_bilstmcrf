from torch.utils.data import Dataset
from .variable import *
from collections import Counter
from tqdm import tqdm
import random

class NERDataset(Dataset):
    def __init__(self, split, tag_to_ix, upsample):
        self.tag_to_ix = tag_to_ix
        self.upsample = upsample
        self.split = split
        if split in ['train', 'dev', 'toy', 'train_']:
            self.text, self.labels = self.load_annotated_data(split)
        elif split == 'test':
            self.text = self.load_test_data(split)
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

        if self.upsample and split == 'train':
            label_counts = Counter([label for sentence_labels in sentence_labels for label in sentence_labels])
            label_factors = {label: 10000 // count for label, count in label_counts.items()}
            for label in label_counts:
                print(f'{label}: {label_counts[label]} ({label_factors[label]}x)')

            upsampled_sentences = []
            upsampled_sentence_labels = []
            # only upsample BME labels
            B = ['B-ORG', 'B-LOC', 'B-PER', 'B-GPE']
            S = ['S-ORG', 'S-LOC', 'S-PER', 'S-GPE']
            for sentence, sentence_label in tqdm(zip(sentences, sentence_labels)):
                if any([label in B+S for label in sentence_label]):
                    factor = max([label_factors[label] for label in sentence_label if label in B+S])
                    mask = [idx for idx, label in enumerate(sentence_label) if label != 'O']
                    sentence = [sentence[idx] for idx in mask]
                    sentence_label = [sentence_label[idx] for idx in mask]
                    upsampled_sentences.extend([sentence] * factor)
                    upsampled_sentence_labels.extend([sentence_label] * factor) 

            upsampled_sentences.extend(sentences)
            upsampled_sentence_labels.extend(sentence_labels)
            label_counts = Counter([label for label in upsampled_sentence_labels for label in label])
            for label in label_counts:
                print(f'{label}: {label_counts[label]}')
            return upsampled_sentences, upsampled_sentence_labels
        return sentences, sentence_labels

    def load_test_data(self, split):
        if split != 'test':
            raise ValueError('split must be test')
        words = []
        with open('data/' + split + '.csv', 'r') as f:
            for i, line in enumerate(f):
                # skip 1st and empty lines
                line = line.strip()
                if line == '' or i == 0:
                    continue
                idx = line.find(',')
                word = line[idx+1:]
                words.append(word)
        
        # split on '。' and '.' and '!' and '?' and '？' and '！'
        splitter = ['。', '.','．', '!', '！', '?', '？']
        sentences = []
        start = 0
        for i in range(len(words)):
            if words[i] in splitter:
                sentence = words[start:i+1]
                start = i+1
                sentences.append(sentence)
            # in case sentence is too long
            if i - start > 128 and words[i] in ['，', ',', '、', '、']:
                sentence = words[start:i+1]
                start = i+1
                sentences.append(sentence)     

        # in case last sentence does not end with a splitter
        if start < len(words):
            sentence = words[start:]
            sentences.append(sentence)

        return sentences
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        if self.split in ['train', 'dev']:
            return self.text[index], self.labels[index]
        elif self.split == 'test':
            return self.text[index]
