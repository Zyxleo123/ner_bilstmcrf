import torch
from torch import nn
from torchcrf import CRF
from transformers import AutoModel
import time
# from .crf import CRF
from ..variable import *

class BiLSTMCRF(nn.Module):

    def __init__(self, label_to_idx, lstm_state_dim, pretrained_model_name='bert-base-chinese', freeze_bert=False):
        super(BiLSTMCRF, self).__init__()
        self.lstm_state_dim = lstm_state_dim
        self.K = len(label_to_idx)
        self.label_to_idx = label_to_idx

        self.bert_like_model = AutoModel.from_pretrained(pretrained_model_name)
        if freeze_bert:
            for param in self.bert_like_model.parameters():
                param.requires_grad = False
        self.embedding_dim = self.bert_like_model.config.hidden_size
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_state_dim // 2, num_layers=1, 
                            bidirectional=True, batch_first=False, bias=False)
        self.projection = nn.Linear(self.lstm_state_dim, self.K)
        self.crf = CRF(self.K, batch_first=False)
        # The `self.set_transitions()` method in the `BiLSTMCRF` class is setting transition scores in
        # the CRF (Conditional Random Field) layer based on the predefined transition rules for the
        # specific named entity recognition task.
        self.set_transitions(-5.0)
        self.dropout = nn.Dropout(0.3)


    def _bilstm(self, input_ids, attention_mask, word_ids):
        start_time = time.time()
        embeds = self.bert_like_model(input_ids, attention_mask=attention_mask).last_hidden_state
        embeds = embeds.permute(1, 0, 2) # change to time-step first
        print(f"Embedding time: {time.time()-start_time:.2f}s")

        feats, self.state = self.lstm(embeds)
        start_time = time.time()
        feats, word_masks = self._char_feat_to_word_feat(feats, word_ids, attention_mask)
        print(f"Convert time: {time.time()-start_time:.2f}s")
        feats = self.dropout(feats)
        emit_scores = self.projection(feats)
        return emit_scores, word_masks

    def _char_feat_to_word_feat(self, char_feats, word_ids, attention_mask):
        # word_feats is the average of char_feats of the same word. The same word is defined by the same word_id.
        # char_feats: L x B x D
        # word_ids: B x L. Looks like: [[-1(CLS), 0, 1, 1, 2, -1(SEP), -1(PAD)], [-1, 0, 1, 1, 2, 2, -1], ...]
        # Output: word_feats: W x B x D, masks: W x B; W is the max word number in a batch; pad word_feats and masks with 0
    
        L, B, D = char_feats.shape
        # word_nums marks how many words each batch has
        word_nums = torch.max(word_ids, dim=1)[0] + 1 # B
        # char_nums marks how many non-special tokens(subwords) each batch has
        char_nums = torch.sum(attention_mask, dim=1) - 2 # take off 2 special tokens of bert; B
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
    
    def _sanity_check(self, word_feats, masks, word_ids):
        L, B, D = word_feats.shape
        word_nums = []
        for batch_word_ids in word_ids:
            word_nums.append(max(batch_word_ids) + 1)
        for b in range(B):
            word_num = word_nums[b]
            assert torch.all(masks[:word_num, b] == 1)
            assert torch.all(masks[word_num:, b] == 0)
            assert torch.all(word_feats[word_num:, b, :] == 0.)

    def forward(self, input_ids, word_ids):
        logits, _ = self._bilstm(input_ids, word_ids)
        return logits
    
    def calculate_loss(self, input_ids, attention_mask, word_ids, labels):
        labels = labels.T
        emit_score, word_masks = self._bilstm(input_ids, attention_mask, word_ids)
        start_time = time.time()
        loss = - self.crf(emit_score, labels, mask=word_masks, reduction='token_mean')
        print(f"CRF forward time: {time.time()-start_time:.2f}s")
        return loss

    def predict(self, input_ids, attention_mask, word_ids):
        emit_scores, word_masks = self._bilstm(input_ids, attention_mask, word_ids)
        start_time = time.time()
        pred = self.crf.decode(emit_scores, mask=word_masks)
        print(f"CRF decode time: {time.time()-start_time:.2f}s")
        return pred

    def set_transitions(self, fill):
        # from S-xxx to M-*/E-*
        for entity in ENTITY_SUB_TYPE:
            self.crf.transitions.data[self.label_to_idx['S-'+entity], self.label_to_idx['M-'+entity]] = fill
            self.crf.transitions.data[self.label_to_idx['S-'+entity], self.label_to_idx['E-'+entity]] = fill

        # from B-xxx to B-*/S-*/M-yyy/E-yyy/O/
        for to_entity in ENTITY_SUB_TYPE:
            for from_entity in ENTITY_SUB_TYPE:
                self.crf.transitions.data[self.label_to_idx['B-'+from_entity], self.label_to_idx['B-'+to_entity]] = fill
                self.crf.transitions.data[self.label_to_idx['B-'+from_entity], self.label_to_idx['S-'+to_entity]] = fill
                if to_entity != from_entity:
                    self.crf.transitions.data[self.label_to_idx['B-'+from_entity], self.label_to_idx['M-'+to_entity]] = fill
                    self.crf.transitions.data[self.label_to_idx['B-'+from_entity], self.label_to_idx['E-'+to_entity]] = fill
                self.crf.transitions.data[self.label_to_idx['B-'+from_entity], self.label_to_idx['O']] = fill

        # from M-xxx to B-*/S-*/M-yyy/E-yyy/O/
        for to_entity in ENTITY_SUB_TYPE:
            for from_entity in ENTITY_SUB_TYPE:
                self.crf.transitions.data[self.label_to_idx['M-'+from_entity], self.label_to_idx['B-'+to_entity]] = fill
                self.crf.transitions.data[self.label_to_idx['M-'+from_entity], self.label_to_idx['S-'+to_entity]] = fill
                if to_entity != from_entity:
                    self.crf.transitions.data[self.label_to_idx['M-'+from_entity], self.label_to_idx['M-'+to_entity]] = fill
                    self.crf.transitions.data[self.label_to_idx['M-'+from_entity], self.label_to_idx['E-'+to_entity]] = fill
                self.crf.transitions.data[self.label_to_idx['M-'+from_entity], self.label_to_idx['O']] = fill

        
        # from E-* to M-*/E-*/
        for to_entity in ENTITY_SUB_TYPE:
            for from_entity in ENTITY_SUB_TYPE:
                self.crf.transitions.data[self.label_to_idx['M-'+from_entity], self.label_to_idx['E-'+to_entity]] = fill
                self.crf.transitions.data[self.label_to_idx['E-'+from_entity], self.label_to_idx['E-'+to_entity]] = fill

        # from O to M-*/E-*
        for entity in ENTITY_SUB_TYPE:
            self.crf.transitions.data[self.label_to_idx['O'], self.label_to_idx['M-'+entity]] = fill
            self.crf.transitions.data[self.label_to_idx['O'], self.label_to_idx['E-'+entity]] = fill

