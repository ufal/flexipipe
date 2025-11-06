#!/usr/bin/env python3
"""
BERT-based text normalizer for historical documents.

Can be trained on (original, normalized) pairs and used as a preprocessing step
before tagging with flexipipe.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import torch
from torch import nn
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    PreTrainedTokenizer, PreTrainedModel
)
from datasets import Dataset, DatasetDict
import numpy as np

# Disable tokenizers parallelism warning
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


@dataclass
class NormalizerConfig:
    """Configuration for the normalizer."""
    bert_model: str = "bert-base-portuguese-cased"  # or "neuralmind/bert-base-portuguese-cased"
    max_length: int = 128  # Shorter for normalization (single words/short phrases)
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 10  # More epochs for normalization
    output_dir: str = "models/normalizer"
    similarity_threshold: float = 0.7  # Fallback similarity threshold


class NormalizerModel(nn.Module):
    """
    BERT-based normalizer that predicts normalized form from original.
    
    Two approaches:
    1. Classification: Predict normalized form from vocabulary (faster, requires vocab)
    2. Generation: Generate normalized form character-by-character (more flexible)
    
    This implements approach 1 (classification) for speed and accuracy.
    """
    
    def __init__(self, base_model_name: str, vocab_size: int, num_labels: int):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.base_model.config.hidden_size
        
        # Classification head: predict normalized form from vocabulary
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {'loss': loss, 'logits': logits}


class Normalizer:
    """BERT-based normalizer for historical documents."""
    
    def __init__(self, config: NormalizerConfig, vocab: Optional[Dict] = None):
        self.config = config
        self.vocab = vocab or {}
        self.tokenizer = None
        self.model = None
        self.normalized_forms = []  # List of normalized forms (labels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def train(self, train_data: List[Tuple[str, str]], dev_data: Optional[List[Tuple[str, str]]] = None):
        """
        Train normalizer on (original, normalized) pairs.
        
        Args:
            train_data: List of (original_form, normalized_form) tuples
            dev_data: Optional dev set for validation
        """
        print(f"Training normalizer on {len(train_data)} examples...")
        
        # Build vocabulary of normalized forms
        normalized_set = set()
        for original, normalized in train_data:
            normalized_set.add(normalized.lower())
        
        if dev_data:
            for original, normalized in dev_data:
                normalized_set.add(normalized.lower())
        
        self.normalized_forms = sorted(normalized_set)
        num_labels = len(self.normalized_forms)
        
        print(f"Found {num_labels} unique normalized forms")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_model)
        self.model = NormalizerModel(self.config.bert_model, len(self.tokenizer), num_labels)
        self.model.to(self.device)
        
        # Prepare datasets
        def tokenize_function(examples):
            # Tokenize original forms
            tokenized = self.tokenizer(
                examples['original'],
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            
            # Convert normalized forms to label indices
            labels = []
            for norm in examples['normalized']:
                norm_lower = norm.lower()
                if norm_lower in self.normalized_forms:
                    labels.append(self.normalized_forms.index(norm_lower))
                else:
                    labels.append(-100)  # Ignore index
            
            return {
                'input_ids': tokenized['input_ids'].squeeze(),
                'attention_mask': tokenized['attention_mask'].squeeze(),
                'labels': torch.tensor(labels)
            }
        
        # Create datasets
        train_dict = {
            'original': [orig for orig, _ in train_data],
            'normalized': [norm for _, norm in train_data]
        }
        train_dataset = Dataset.from_dict(train_dict)
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        
        if dev_data:
            dev_dict = {
                'original': [orig for orig, _ in dev_data],
                'normalized': [norm for _, norm in dev_data]
            }
            dev_dataset = Dataset.from_dict(dev_dict)
            dev_dataset = dev_dataset.map(tokenize_function, batched=True)
        else:
            dev_dataset = None
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=500,
            logging_steps=100,
            eval_strategy="epoch" if dev_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if dev_dataset else False,
        )
        
        # Custom trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
        )
        
        # Train
        trainer.train()
        
        # Save model
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save normalization vocabulary
        with open(Path(self.config.output_dir) / 'normalized_forms.json', 'w', encoding='utf-8') as f:
            json.dump(self.normalized_forms, f, ensure_ascii=False, indent=2)
        
        print(f"Model saved to {self.config.output_dir}")
    
    def normalize(self, word: str, fallback_to_similarity: bool = True) -> Optional[str]:
        """
        Normalize a word using the trained model.
        
        Args:
            word: Word to normalize
            fallback_to_similarity: If True, fall back to similarity matching if model fails
        
        Returns:
            Normalized form if found, None otherwise
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize
        tokenized = self.tokenizer(
            word,
            padding='max_length',
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            predicted_idx = torch.argmax(logits, dim=-1).item()
        
        # Get normalized form
        if predicted_idx < len(self.normalized_forms):
            normalized = self.normalized_forms[predicted_idx]
            return normalized
        
        # Fallback to similarity matching if enabled
        if fallback_to_similarity and self.vocab:
            from flexipipe import find_similar_words
            similar = find_similar_words(word, self.vocab, threshold=self.config.similarity_threshold)
            if similar:
                return similar[0][0]
        
        return None
    
    def load_model(self, model_dir: Path):
        """Load a trained normalizer model."""
        model_dir = Path(model_dir)
        
        # Load normalized forms
        with open(model_dir / 'normalized_forms.json', 'r', encoding='utf-8') as f:
            self.normalized_forms = json.load(f)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load model (use from_pretrained if available, otherwise load state_dict)
        try:
            # Try to load as a saved model
            self.model = NormalizerModel(
                self.config.bert_model,
                len(self.tokenizer),
                len(self.normalized_forms)
            )
            # Try to load state dict
            state_dict_path = model_dir / 'pytorch_model.bin'
            if state_dict_path.exists():
                self.model.load_state_dict(torch.load(state_dict_path, map_location=self.device))
            else:
                # Try to load from model files
                from transformers import AutoModelForSequenceClassification
                base_model = AutoModelForSequenceClassification.from_pretrained(model_dir, map_location=self.device)
                # Extract the classifier weights if available
                if hasattr(base_model, 'classifier'):
                    self.model.classifier.load_state_dict(base_model.classifier.state_dict())
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}. Model will be initialized randomly.", file=sys.stderr)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded normalizer from {model_dir}")


def load_normalization_pairs(file_path: Path) -> List[Tuple[str, str]]:
    """
    Load normalization pairs from file.
    
    Expected format (one pair per line):
    original\tnormalized
    or
    original,normalized
    """
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Try tab-separated first
            if '\t' in line:
                parts = line.split('\t', 1)
            elif ',' in line:
                parts = line.split(',', 1)
            else:
                continue
            
            if len(parts) == 2:
                original, normalized = parts
                pairs.append((original.strip(), normalized.strip()))
    
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='BERT-based text normalizer',
        prog='flexipipe normalizer'
    )
    subparsers = parser.add_subparsers(dest='mode', help='Mode')
    
    # Train mode
    train_parser = subparsers.add_parser('train', help='Train normalizer')
    train_parser.add_argument('--train-data', type=Path, required=True,
                             help='Training data file (original\\tnormalized pairs)')
    train_parser.add_argument('--dev-data', type=Path,
                             help='Dev data file (optional)')
    train_parser.add_argument('--bert-model', default='neuralmind/bert-base-portuguese-cased',
                             help='BERT base model')
    train_parser.add_argument('--output-dir', type=Path, default=Path('models/normalizer'),
                             help='Output directory')
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--learning-rate', type=float, default=2e-5)
    train_parser.add_argument('--num-epochs', type=int, default=10)
    
    # Normalize mode
    norm_parser = subparsers.add_parser('normalize', help='Normalize text')
    norm_parser.add_argument('--model', type=Path, required=True,
                            help='Path to trained model')
    norm_parser.add_argument('--input', type=Path, required=True,
                            help='Input file (one word per line)')
    norm_parser.add_argument('--output', type=Path,
                            help='Output file (default: stdout)')
    norm_parser.add_argument('--vocab', type=Path,
                            help='Vocabulary file for fallback similarity matching')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Load training data
        train_pairs = load_normalization_pairs(args.train_data)
        dev_pairs = None
        if args.dev_data:
            dev_pairs = load_normalization_pairs(args.dev_data)
        
        # Train
        config = NormalizerConfig(
            bert_model=args.bert_model,
            output_dir=str(args.output_dir),
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs
        )
        normalizer = Normalizer(config)
        normalizer.train(train_pairs, dev_pairs)
    
    elif args.mode == 'normalize':
        # Load model
        vocab = {}
        if args.vocab:
            with open(args.vocab, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
        
        config = NormalizerConfig()
        normalizer = Normalizer(config, vocab)
        normalizer.load_model(args.model)
        
        # Normalize
        with open(args.input, 'r', encoding='utf-8') as f_in:
            output_file = open(args.output, 'w', encoding='utf-8') if args.output else sys.stdout
            for line in f_in:
                word = line.strip()
                if not word:
                    output_file.write('\n')
                    continue
                
                normalized = normalizer.normalize(word)
                if normalized:
                    output_file.write(f"{word}\t{normalized}\n")
                else:
                    output_file.write(f"{word}\t{word}\n")
            
            if args.output:
                output_file.close()
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

