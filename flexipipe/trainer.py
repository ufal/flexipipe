"""
Trainer module for FlexiPipe.
"""
import torch
from torch import nn
from transformers import Trainer, TrainingArguments

class MultiTaskTrainer(Trainer):
    """Custom trainer for multi-task learning."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels_upos = inputs.pop("labels_upos", None)
        labels_xpos = inputs.pop("labels_xpos", None)
        labels_feats = inputs.pop("labels_feats", None)
        labels_lemma = inputs.pop("labels_lemma", None)
        labels_norm = inputs.pop("labels_norm", None)
        labels_head = inputs.pop("labels_head", None)
        labels_deprel = inputs.pop("labels_deprel", None)
        
        outputs = model(**inputs, labels_upos=labels_upos, labels_xpos=labels_xpos, 
                       labels_feats=labels_feats, labels_lemma=labels_lemma,
                       labels_norm=labels_norm,
                       labels_head=labels_head, labels_deprel=labels_deprel)
        loss = outputs.get('loss')
        
        return (loss, outputs) if return_outputs else loss



