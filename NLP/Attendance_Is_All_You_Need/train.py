import math
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from datetime import datetime
from DRdataset import load_noisy_dataset, load_sts_dataset, load_chapter_dataset
from torch.utils.data import DataLoader
import sys
import os
import gzip
import csv
import torch

# Config
pdf_path = '/home/aiku/AIKU/nlp_team1/document_retrieval/Project.pdf'
sts_path = 'datasets/stsbenchmark.tsv.gz'
chapter_path = '/home/aiku/AIKU/nlp_team1/document_retrieval/pretrain.csv'
model_name = 'sentence-transformers/all-mpnet-base-v2'
batch_size = 16
pretrain_epochs = 5
finetune_epochs = 5
evaluation_steps = 100
experiment_name = f'noisy-{pretrain_epochs}-sts-{finetune_epochs}.pth'

# Model
print('Load Model')
word_embedding_model = models.Transformer(model_name)

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Dataset
print('Load Chapter Dataset for Pre-training')
chapter_tds, chapter_eds = load_chapter_dataset(chapter_path)
chapter_dataloader = DataLoader(chapter_tds, batch_size, True)
chapter_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(chapter_eds, name='chapter-eval')

print('Load Noisy Dataset for Pre-training')
noisy_tds, noisy_eds = load_noisy_dataset(pdf_path, 0.8)
noisy_dataloader = DataLoader(noisy_tds, batch_size, True)
noisy_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(noisy_eds, name='noisy-eval')

print('Load Fine-Tuning Dataset')
sts_tds, sts_eds, sts_testds = load_sts_dataset(sts_path)
sts_dataloader = DataLoader(sts_tds, batch_size, True)
sts_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(sts_eds, name='sts-eval')
sts_testor = EmbeddingSimilarityEvaluator.from_input_examples(sts_testds, name='sts-test')

# Loss
train_loss = losses.CosineSimilarityLoss(model)

# Chapter Pre-training
print('Chapter Pre-Training')
warmup_steps = math.ceil(len(chapter_dataloader) * pretrain_epochs * 0.1)
model.fit(train_objectives=[(chapter_dataloader, train_loss)],
          evaluator=chapter_evaluator,
          evaluation_steps=evaluation_steps,
          epochs=pretrain_epochs,
          warmup_steps=warmup_steps)

# Noisy Pre-training
print('Noisy Pre-Training')
warmup_steps = math.ceil(len(noisy_dataloader) * pretrain_epochs * 0.1)
model.fit(train_objectives=[(noisy_dataloader, train_loss)],
          evaluator=noisy_evaluator,
          evaluation_steps=evaluation_steps,
          epochs=pretrain_epochs,
          warmup_steps=warmup_steps)

# Fine-tuning
print('STS Fine-Tuning')
warmup_steps = math.ceil(len(sts_dataloader) * finetune_epochs * 0.1)
model.fit(train_objectives=[(sts_dataloader, train_loss)],
          evaluator=sts_evaluator,
          evaluation_steps=evaluation_steps,
          epochs=finetune_epochs,
          warmup_steps=warmup_steps)

# Test
print('Test')
sts_testor(model, output_path=f'/home/aiku/AIKU/nlp_team1/document_retrieval/results/{experiment_name}.csv')

torch.save(model.state_dict(), f'/home/aiku/AIKU/nlp_team1/document_retrieval/model_weights/{experiment_name}.pth')