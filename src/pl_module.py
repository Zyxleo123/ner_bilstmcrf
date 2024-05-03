import torch
import time
from pytorch_lightning import LightningModule
from .models import BiLSTMCRF
from .variable import LABELS, IDX_TO_LABEL

class LightningBiLSTMCRF(LightningModule):
    def __init__(self, label_to_idx, lstm_layer_num, lstm_state_dim, char_level,
                bert_lr, lstm_lr, crf_lr, optimizer, pretrained_model_name, freeze_bert):
        super(LightningBiLSTMCRF, self).__init__()
        self.model = BiLSTMCRF(label_to_idx, lstm_layer_num, lstm_state_dim, char_level, pretrained_model_name, freeze_bert)
        self.bert_lr = bert_lr
        self.lstm_lr = lstm_lr
        self.crf_lr = crf_lr
        self.char_level = char_level
        self.optimizer = optimizer
        self.val_gts = []
        self.val_preds = []

    def forward(self, input_ids, word_ids):
        return self.model(input_ids, word_ids)

    def training_step(self, batch, batch_idx):
        start_time = time.time()
        loss = self.model.calculate_loss(**batch)
        print(f"Training step time: {time.time()-start_time:.2f}s")
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, word_ids, gt = batch['input_ids'], batch['attention_mask'], batch['word_ids'], batch['labels']
        pred = self.model.predict(input_ids=input_ids, attention_mask=attention_mask, word_ids=word_ids)
        if not self.char_level:
            pred = [[IDX_TO_LABEL[int(y)] for y in b] for b in pred]
            gt = [[IDX_TO_LABEL[int(y)] for y in b] for b in gt]
        else:
            pred = [[IDX_TO_LABEL[int(y)] for y in b[1:-1]] for b in pred]
            gt = [[IDX_TO_LABEL[int(y)] for y in b[1:-1]] for b in gt]
        self.val_gts.extend(gt)
        self.val_preds.extend(pred)
        return gt, pred

    def on_validation_epoch_end(self):
        report = classification_report(self.val_gts, self.val_preds, LABELS)
        self.log_dict(report, prog_bar=True)
        self.val_gts.clear()
        self.val_preds.clear()
        
    def configure_optimizers(self):
        # apply smaller learning rate to the bert model
        bert_params = self.model.bert_like_model.parameters()
        crf_params = self.model.crf.parameters()
        lstm_params = [p for p in self.model.parameters() if id(p) not in map(id, bert_params) and id(p) not in map(id, crf_params)]
        
        if self.optimizer == 'adam':
            optimizer_type = torch.optim.Adam
        elif self.optimizer == 'sgd':
            optimizer_type = torch.optim.SGD
        elif self.optimizer == 'adamw':
            optimizer_type = torch.optim.AdamW
        else:
            raise ValueError('optimizer must be adam or sgd or adamw')

        momentum = {'momentum': 0.9} if self.optimizer == 'sgd' else {}
        if self.bert_lr == 0.:
            optimizer = optimizer_type([
                {'params': crf_params, 'lr': self.crf_lr}, 
                {'params': lstm_params, 'lr': self.lstm_lr},
            ], lr=self.lstm_lr, **momentum)
        else:
            optimizer = optimizer_type([
                {'params': bert_params, 'lr': self.bert_lr},
                {'params': crf_params, 'lr': self.crf_lr}, 
                {'params': lstm_params, 'lr': self.lstm_lr},
            ], lr=self.lstm_lr, **momentum)
        return optimizer

def classification_report(gt, pred, label_set):
    # return a dict that report: 
    # {'label1': {'precision': 0.5, 'recall': 0.5, 'f1-score': 0.5, 'support': 1}, ..., 'macro avg': {'precision': 0.5, 'recall': 0.5, 'f1-score': 0.5, 'support': 1}, 'weighted avg': {'precision': 0.5, 'recall': 0.5, 'f1-score': 0.5, 'support': 1}}
    report = {}
    for label in label_set:
        report[label] = {}
        tp = fp = fn = 0
        for g, p in zip(gt, pred):
            for gg, pp in zip(g, p):
                if gg == label and pp == label:
                    tp += 1
                elif gg == label and pp != label:
                    fn += 1
                elif gg != label and pp == label:
                    fp += 1
        report[label]['precision'] = tp / (tp + fp) if tp + fp != 0 else 0
        report[label]['recall'] = tp / (tp + fn) if tp + fn != 0 else 0
        report[label]['f1-score'] = 2 * report[label]['precision'] * report[label]['recall'] / (report[label]['precision'] + report[label]['recall']) if report[label]['precision'] + report[label]['recall'] != 0 else 0
        report[label]['support'] = tp + fn
    macro_avg = {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}
    for label in label_set:
        macro_avg['precision'] += report[label]['precision']
        macro_avg['recall'] += report[label]['recall']
        macro_avg['f1-score'] += report[label]['f1-score']
        macro_avg['support'] += report[label]['support']
    macro_avg['precision'] /= len(label_set)
    macro_avg['recall'] /= len(label_set)
    macro_avg['f1-score'] /= len(label_set)
    report['macro avg'] = macro_avg
    # pop precision & recall in every label
    for label in label_set:
        report[label].pop('precision')
        report[label].pop('recall')
    report = dict_flatten(report)
    return report

def dict_flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(dict_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
