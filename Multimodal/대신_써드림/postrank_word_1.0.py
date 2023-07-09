import os
import json
import math
import networkx as nx
from tqdm import tqdm
from datetime import date, timedelta
from collections import Counter
from collections import defaultdict
from typing import List, Dict, Tuple, Union, Callable, Mapping

import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
from scipy.sparse import csr_matrix
from konlpy.tag import Okt

import pandas as pd
import sys


class TextRankByWord():
    def __init__(self):
        self.sentences = None
        self.rank = None
        self.graph_matrix = None
        self.personalization = None
        self.vocab2idx = None
        self.idx2vocab = None


    def _getVocabularyInfo(self,
                           tokenizer: Callable,
                           min_word_count: int = 2):
        assert 0 < min_word_count, 'Minimum word count should be a positive integer.'
        print(f"Processing... Vocabulary Information", end=" ")
        counter = Counter(word for sentence in self.sentences for word in tokenizer(sentence))
        counter = {word: count for word, count in counter.items() if count >= min_word_count}
        idx2vocab = [w for w, _ in sorted(counter.items(), key=lambda x: -x[1])]
        vocab2idx = {vocab:idx for idx, vocab in enumerate(idx2vocab)}
        print(f"-> Done.")
        return idx2vocab, vocab2idx


    def _getSparseMatrix(self,
                           counter: dict,
                           matrix_size: int) -> scipy.sparse.spmatrix:
        data, rows, columns,  = [], [], []
        for (row, column), key in counter.items():
            data.append(key)
            rows.append(row)
            columns.append(column)
            
        return csr_matrix((data, (rows, columns)), shape=(matrix_size, matrix_size))

    
    def _getTokenizedSentences(self,
                               tokenizer: Callable) -> List:
        return [tokenizer(sentence) for sentence in self.sentences]


    def _getGraphMatrix(self,
                        tokenizer: callable,
                        window_size: int = 2,
                        min_cooccurrence: int = 2) -> scipy.sparse.spmatrix:
        assert 0 <= window_size, 'Size of window should be a positive integer or 0.'
        assert 0 < min_cooccurrence, 'Min_cooccurrence should be a positive integer.'
        print(f"Processing... Graph Matrix", end=" ")
        #1 Get tokenized sentences
        tokenized_sentences = self._getTokenizedSentences(tokenizer)
        #2 Get co-occurence matrix in dictionary form
        counter = defaultdict(int)
        for tokenized_sentence in tokenized_sentences:
            for i, center_token in enumerate(tokenized_sentence):
                if window_size == 0:
                    s_bound, e_bound = 0, len(tokenized_sentence)
                else:
                    s_bound = max(0, i - window_size)
                    e_bound = min(i + window_size, len(tokenized_sentence))
                for j in range(s_bound, e_bound):
                    if i == j:
                        continue
                    try:
                        counter[(self.vocab2idx[center_token], self.vocab2idx[tokenized_sentence[j]])] += 1
                        counter[(self.vocab2idx[tokenized_sentence[j]], self.vocab2idx[center_token])] += 1
                    except:
                        continue
        #3 Clip keys for which the co-occurence count is less than or equal to the reference value.
        counter = {k:v for k,v in counter.items() if v >= min_cooccurrence}
        #4 Convert to CSR matrix.
        matrix_size = len(self.vocab2idx)
        graph_matrix = self._getSparseMatrix(counter, matrix_size)
        print(f"-> Done.")
        return graph_matrix


    def _updateRank(self,
                    damping: float = 0.85,
                    personalization: Mapping = None, 
                    max_iter: int = 100,
                    tol: float = 0.000001) -> npt.ArrayLike:
        assert 0 < damping < 1, 'damping should be between 0 and 1.'
        assert max_iter > 0, 'max_iter should be a positive integer.'
        print(f"Processing... TextRank", end=" ")
        if personalization:
            bias = personalization
        else:
            bias = None
        G = nx.from_numpy_array(self.graph_matrix)
        rank  = np.array(list(nx.pagerank(G=G,
                                          alpha=damping, 
                                          personalization=bias,
                                          max_iter = max_iter,
                                          tol = tol).values()))
        print(f"-> Done.")
        return rank
    

    def getKeyWords(self,
                    k: int = 30):
        """Get top k keywords
        """
        if k > len(self.rank):
            k = len(self.rank)
        keyword_idxs = self.rank.argsort()[-k:]
        keywords = [(self.idx2vocab[idx], self.rank[idx]) for idx in reversed(keyword_idxs)]
        return keywords


    def analyze(self,
                sentences: List,
                tokenizer: Callable,
                damping: float = 0.85,
                max_iter: int = 100,
                tol: float = 0.00001,
                min_word_count: int = 2,
                window_size: int = 2,
                min_cooccurrence: int = 2):
        """Calculates Textrank for sentences given as input, and stores them.
        """
        self.sentences = sentences
        self.idx2vocab, self.vocab2idx = self._getVocabularyInfo(tokenizer=tokenizer,
                                                                 min_word_count=min_word_count)
        self.graph_matrix = self._getGraphMatrix(tokenizer=tokenizer,
                                                 window_size=window_size,
                                                 min_cooccurrence=min_cooccurrence)
        self.rank = self._updateRank(damping=damping,
                                     personalization=None,
                                     max_iter=max_iter,
                                     tol=tol)
        print(f"! Process has been successfully completed.")

if __name__ == "__main__":
    FILE_DIR = "result"
    SIZE = 256
    # STD_TIME = date.today()
    # STD_TIME = date.fromisoformat('2018-01-01')
    arguments = sys.argv
    STD_TIME = date.fromisoformat(sys.argv[1])

    df = pd.DataFrame()
    posts = []
    
    # while (date.today() - STD_TIME).days >= 0:
    #     count = 0
        
    #     for file_name in reversed(sorted(os.listdir(FILE_DIR))[1:]):
    #         if file_name.endswith('.json'):
    #             with open(os.path.join(FILE_DIR, file_name)) as post:
    #                 post = json.load(post)
    #                 if (STD_TIME - date.fromisoformat(post['time'].split()[0])).days >= 0:
    #                     posts.append(post['title'])
    #                     count += 1
    #             if count >= SIZE:
    #                 break
    #     STD_TIME = STD_TIME + timedelta(weeks=1)
    
    
    for file_name in reversed(sorted(os.listdir(FILE_DIR))[1:]):
        if file_name.endswith('.json'):
            with open(os.path.join(FILE_DIR, file_name)) as post:
                post = json.load(post)
                day_dff = (date.fromisoformat(post['time'].split()[0]) - STD_TIME).days
                if day_dff > 7:
                    continue
                elif day_dff <= 7 and day_dff >= 0:
                    posts.append(post['title'])
                else:
                    break
        
    okt = Okt()
    def tokenizer(sent):
        words = okt.pos(sent, join=True)
        words = list(filter(lambda x: len(x) > 1, [w.split('/')[0] for w in words if ('/Noun' in w )])) #or '/Adjective' in w or '/Adverb' in w or '/Verb' in w
        return words

    Ranker = TextRankByWord()
    Ranker.analyze(sentences=posts,
                tokenizer=tokenizer, 
                damping=0.85)
    keywords = Ranker.getKeyWords(20)
    
    print('-' * 85)
    for i, (word, rank) in enumerate(keywords):
        print(f"# {i:3} | Word : {word:<20} | Rank : {rank:<20}")
        print('-' * 85)

    for i, (word, rank) in enumerate(keywords):
        df = pd.concat([df, pd.DataFrame([i],
                                         index = [[STD_TIME], [word]])])
    df.to_csv('text_rank.csv', index=True)

    
