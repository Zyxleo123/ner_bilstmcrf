import torch
import time
from pytorch_lightning import LightningModule
from transformers import get_linear_schedule_with_warmup
from .models import BiLSTMCRF
from .variable import LABELS, IDX_TO_LABEL, get_run_name

class LightningBiLSTMCRF(LightningModule):
    def __init__(self, label_to_idx, lstm_layer_num, lstm_state_dim,
                bert_lr, lstm_lr, crf_lr, optimizer, scheduler,
                pretrained_model_name, freeze_bert,
                epochs, steps_per_epoch):
        super(LightningBiLSTMCRF, self).__init__()
        self.model = BiLSTMCRF(label_to_idx, lstm_layer_num, lstm_state_dim, pretrained_model_name, freeze_bert)
        self.bert_lr = bert_lr
        self.lstm_lr = lstm_lr
        self.crf_lr = crf_lr
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.freeze_bert = freeze_bert
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.val_gts = []
        self.val_preds = []
        self.val_losses = []

        self.test_preds = []
        self.save_hyperparameters()

    def forward(self, input_ids, word_ids):
        return self.model(input_ids, word_ids)

    def training_step(self, batch, batch_idx):
        start_time = time.time()
        loss = self.model.calculate_loss(**batch)
        print(f"Training step time: {time.time()-start_time:.2f}s")
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
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
    
    def on_validation_epoch_end(self):
        report = classification_report(self.val_gts, self.val_preds, LABELS)
        self.log_dict(report, prog_bar=True)
        self.log('val_f1', report['macro avg_f1-score'], prog_bar=True)
        avg_loss = sum(self.val_losses) / len(self.val_losses)
        self.log('val_loss', avg_loss, prog_bar=True)
        self.val_gts.clear()
        self.val_preds.clear()
        self.val_losses.clear()
        
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
            f.write('expected\n')
            for p in self.test_preds:
                f.write(f'{p}\n')

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
        if self.scheduler == 'anneal':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
            return [optimizer], [scheduler]
        elif self.scheduler == 'onecycle':
            if not self.freeze_bert:
                raise ValueError('Cannot do one cycle lr scheduler when not freezing bert')
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lstm_lr, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch)
            return [optimizer], [scheduler]
        elif self.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=self.epochs*self.steps_per_epoch)
            return [optimizer], [scheduler]
        else:
            return optimizer

def classification_report(gt, pred, label_set):
    label_wise_metrics = {}
    for label in label_set:
        label_wise_metrics[label] = {}
        tp = fp = fn = 0
        for g, p in zip(gt, pred):
            for gg, pp in zip(g, p):
                if gg == label and pp == label:
                    tp += 1
                elif gg == label and pp != label:
                    fn += 1
                elif gg != label and pp == label:
                    fp += 1
        label_wise_metrics[label]['precision'] = tp / (tp + fp) if tp + fp != 0 else 0
        label_wise_metrics[label]['recall'] = tp / (tp + fn) if tp + fn != 0 else 0
        label_wise_metrics[label]['f1-score'] = 2 * label_wise_metrics[label]['precision'] * label_wise_metrics[label]['recall'] / (label_wise_metrics[label]['precision'] + label_wise_metrics[label]['recall']) if label_wise_metrics[label]['precision'] + label_wise_metrics[label]['recall'] != 0 else 0
        label_wise_metrics[label]['support'] = tp + fn

    macro_metrics = {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}
    for label in label_set:
        macro_metrics['precision'] += label_wise_metrics[label]['precision']
        macro_metrics['recall'] += label_wise_metrics[label]['recall']
        macro_metrics['f1-score'] += label_wise_metrics[label]['f1-score']
        macro_metrics['support'] += label_wise_metrics[label]['support']
    macro_metrics['precision'] /= len(label_set)
    macro_metrics['recall'] /= len(label_set)
    macro_metrics['f1-score'] /= len(label_set)
    
    label_wise_metrics['macro avg'] = macro_metrics
    for label in label_set:
        label_wise_metrics[label].pop('precision')
        label_wise_metrics[label].pop('recall')
    label_wise_metrics = dict_flatten(label_wise_metrics)
    return label_wise_metrics

def dict_flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(dict_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
