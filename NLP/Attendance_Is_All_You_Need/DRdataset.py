from torch.utils.data import Dataset
from sentence_transformers import InputExample, util
import numpy as np
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('punkt')
import utils
import re
import random
import os
import gzip
import csv

# Noisy Dataset

def load_noisy_dataset(pdf_file_path, train_ratio):
    book = utils.read_pdf(pdf_file_path)

    chapter_list = []
    chapter = []

    q = re.compile('\d{1,2}CHAPTER')

    for page in book:
        m = q.search(page)
        if m:
            chapter_list.append(chapter)
            chapter = []
        chapter.append(page)
    
    chapter_list.append(chapter)

    del chapter_list[0]

    for i in range(len(chapter_list)):
        chapter_list[i] = utils.preprocessing(chapter_list[i])
        chapter_list[i] = utils.remove_enter(chapter_list[i])
        chapter_list[i] = utils.remove_problems(chapter_list[i])
        while not (chapter_list[i][0][0][0]).isupper() :
            chapter_list[i][0][0] = chapter_list[i][0][0][1:]

    sentences = []

    for chapter in chapter_list:
        for topic in chapter:
            for sentence in topic:
                sentences.append(sentence)
    
    random.shuffle(sentences)
    train_split = int(len(sentences) * train_ratio)
    
    return (
        NoisyDataset(sentences[:train_split]),
        NoisyDataset(sentences[train_split:]),
    )

class NoisyDataset(Dataset):
    def __init__(self, sentences):
        super().__init__()
        self.sentences = sentences

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        noisy_sent = self.add_noise(sent)
        return InputExample(texts=[noisy_sent, sent], label=1.0)
    
    def __len__(self):
        return len(self.sentences)
    
    def add_noise(self, text, del_ratio=0.6):
        words = nltk.word_tokenize(text)
        n = len(words)
        if n == 0:
            return text
        
        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0:
            keep_or_not[np.random.choice(n)] = True
        words_processed = TreebankWordDetokenizer().detokenize(np.array(words)[keep_or_not])
        return words_processed

# STS Dataset
def load_sts_dataset(sts_path):
    if not os.path.exists(sts_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_path)
    
    train_samples = []
    dev_samples = []
    test_samples = []
    with gzip.open(sts_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row['score']) / 5.0
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

            if row['split'] == 'dev':
                dev_samples.append(inp_example)
            elif row['split'] == 'test':
                test_samples.append(inp_example)
            else:
                train_samples.append(inp_example)
    
    return (
        train_samples,
        dev_samples,
        test_samples,
    )

# Chapter Dataset
def load_chapter_dataset(chapter_path):
    samples = []

    with open(chapter_path, 'r') as f:
        reader = csv.DictReader(f)
        for sample in reader:
            inp_example = InputExample(texts=[sample['sentence1'], sample['sentence2']], label=float(sample['score']))
            samples.append(inp_example)
    
    random.shuffle(samples)
    train_samples = samples[:205000]
    dev_samples = samples[205000:]

    return (
        train_samples,
        dev_samples,
    )