import torch
from torch import nn
from ..variable import *

def log_sum_exp(vec):
    B, K = vec.shape
    max_score = vec[range(B), torch.argmax(vec, dim=1)] # B
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score.view(-1, 1)), dim=1)) # B

class CRF(nn.Module):
    def __init__(self, label_to_idx):
        super(CRF, self).__init__()
        self.label_to_idx = label_to_idx
        self.K = len(label_to_idx)
        self.transitions = nn.Parameter(torch.randn(self.K, self.K))
        # self.freeze_transitions()
    
    def _cal_partition(self, emit_scores, masks):
        L, B = masks.shape
        # \alpha[0, k] = start\_score[k] \\
        forward_var_0 = torch.full((B, self.K), -10000., device=emit_scores.device)
        forward_var_0[:, self.label_to_idx[START_LABEL]] = 0.
        forward_var_t = forward_var_0
        # DP loop
        for t, emit_score in enumerate(emit_scores):
            forward_var_t_k = []  # The working forward variable at this timestep; B x K
            # Populate the next forward_var iteratively in K steps
            for k in range(self.K): # \alpha[t, k] = Logsumexp_{k'}(\alpha[t-1, k'] + emit\_score_t[k] + transition\_score[k', k])
                emit_score = emit_score[range(B), k].view(-1, 1).expand(B, self.K) # B -> B x K
                trans_score = self.transitions[k].view(1, -1).expand(B, self.K) # K -> B x K
                next_label_var = forward_var_t + (emit_score + trans_score) * masks[t].view(-1, 1) # B x K
                # Just keep the previous forward_var if the mask is 0; else sum the path scores
                next_label_var = torch.where(masks[t].bool(), log_sum_exp(next_label_var), forward_var_t[:, k]) # B
                forward_var_t_k.append(next_label_var)
            forward_var_t = torch.stack(forward_var_t_k, dim=1) # B x K
        forward_var_T = forward_var_t + self.transitions[self.label_to_idx[STOP_LABEL]]
        partition = log_sum_exp(forward_var_T)
        return partition
    
    def _cal_sent_score(self, emit_scores, labels, masks):
        L, B = masks.shape
        assert emit_scores.shape == (L, B, self.K)
        assert labels.shape == (L, B)
        score = torch.zeros(B, device=emit_scores.device) # B
        labels = torch.cat([torch.full((1, B), self.label_to_idx[START_LABEL], dtype=torch.long, device=emit_scores.device),
                            labels], dim=0) # Add a [Start] tag; (L + 1) x B
        for t, emit_score in enumerate(emit_scores):
            score += (self.transitions[labels[t + 1], labels[t]] + emit_score[range(B), labels[t + 1]]) * masks[t] # (B + B) x B -> B
        # Add the last transition to STOP_LABEL
        score += self.transitions[self.label_to_idx[STOP_LABEL], labels[torch.sum(masks, dim=0).long(), range(B)]]
        return score

    def _predict(self, emit_scores, masks):
        L, B = masks.shape
        assert emit_scores.shape == (L, B, self.K)
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
                assert next_label_var.shape == (B, self.K)
                best_label_ids = torch.argmax(next_label_var, dim=1) # B
                assert best_label_ids.shape == (B,)
                next_label_var = torch.where(masks[t].bool(), next_label_var[range(B), best_label_ids], forward_var_t[:, next_label]) # B
                bptrs_t.append(best_label_ids)
                forward_var_t_k.append(next_label_var)
                assert bptrs_t[-1].shape == forward_var_t_k[-1].shape == (B,)
            # Now add in the emission scores, and assign forward_var to the set of viterbi variables we just computed
            forward_var_t = torch.stack(forward_var_t_k, dim=1) + emit_score * masks[t].view(-1, 1) # B x K
            bptrs.append(torch.stack(bptrs_t, dim=1)) # B x K

        bptrs = torch.stack(bptrs, dim=0) # L x B x K
        assert bptrs.shape == (L, B, self.K)
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
            assert start == self.label_to_idx[START_LABEL]
            best_path.reverse()
            assert len(best_path) == torch.sum(masks[:, b]).long()
            best_paths.append(best_path)
        return best_scores, best_paths

    def freeze_transitions(self):
        # freeze from S-xxx to M-*/E-*
        for entity in ENTITY_SUB_TYPE:
            self.transitions.data[self.label_to_idx['M-'+entity], self.label_to_idx['S-'+entity]] = -10000.
            self.transitions.data[self.label_to_idx['E-'+entity], self.label_to_idx['S-'+entity]] = -10000.

        # freeze from B-xxx to B-*/S-*/M-yyy/E-yyy/O/
        for from_entity in ENTITY_SUB_TYPE:
            for to_entity in ENTITY_SUB_TYPE:
                self.transitions.data[self.label_to_idx['B-'+to_entity], self.label_to_idx['B-'+from_entity]] = -10000.
                self.transitions.data[self.label_to_idx['S-'+to_entity], self.label_to_idx['B-'+from_entity]] = -10000.
                if from_entity != to_entity:
                    self.transitions.data[self.label_to_idx['M-'+to_entity], self.label_to_idx['B-'+from_entity]] = -10000.
                    self.transitions.data[self.label_to_idx['E-'+to_entity], self.label_to_idx['B-'+from_entity]] = -10000.
                self.transitions.data[self.label_to_idx['O'], self.label_to_idx['B-'+from_entity]] = -10000.

        # freeze from M-xxx to B-*/S-*/M-yyy/E-yyy/O/
        for from_entity in ENTITY_SUB_TYPE:
            for to_entity in ENTITY_SUB_TYPE:
                self.transitions.data[self.label_to_idx['B-'+to_entity], self.label_to_idx['M-'+from_entity]] = -10000.
                self.transitions.data[self.label_to_idx['S-'+to_entity], self.label_to_idx['M-'+from_entity]] = -10000.
                if from_entity != to_entity:
                    self.transitions.data[self.label_to_idx['M-'+to_entity], self.label_to_idx['M-'+from_entity]] = -10000.
                    self.transitions.data[self.label_to_idx['E-'+to_entity], self.label_to_idx['M-'+from_entity]] = -10000.
                self.transitions.data[self.label_to_idx['O'], self.label_to_idx['M-'+from_entity]] = -10000.
        
        # freeze from E-* to M-*/E-*/
        for from_entity in ENTITY_SUB_TYPE:
            for to_entity in ENTITY_SUB_TYPE:
                self.transitions.data[self.label_to_idx['M-'+to_entity], self.label_to_idx['E-'+from_entity]] = -10000.
                self.transitions.data[self.label_to_idx['E-'+to_entity], self.label_to_idx['E-'+from_entity]] = -10000.

        # freeze from O to M-*/E-*
        for entity in ENTITY_SUB_TYPE:
            self.transitions.data[self.label_to_idx['M-'+entity], self.label_to_idx['O']] = -10000.
            self.transitions.data[self.label_to_idx['E-'+entity], self.label_to_idx['O']] = -10000.
        
        # freeze from START_LABEL to M-*/E-*
        for entity in ENTITY_SUB_TYPE:
            self.transitions.data[self.label_to_idx['M-'+entity], self.label_to_idx[START_LABEL]] = -10000.
            self.transitions.data[self.label_to_idx['E-'+entity], self.label_to_idx[START_LABEL]] = -10000.
        
        # freeze from B-*/M-* to STOP_LABEL
        for entity in ENTITY_SUB_TYPE:
            self.transitions.data[self.label_to_idx[STOP_LABEL], self.label_to_idx['B-'+entity]] = -10000.
            self.transitions.data[self.label_to_idx[STOP_LABEL], self.label_to_idx['M-'+entity]] = -10000. 

        self.transitions.data[self.label_to_idx[START_LABEL], :] = -10000.
        self.transitions.data[:, self.label_to_idx[STOP_LABEL]] = -10000.
        self.transitions.data[self.label_to_idx[PAD_LABEL], :] = -10000.
        self.transitions.data[:, self.label_to_idx[PAD_LABEL]] = -10000.

