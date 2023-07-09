import sys
import json
import os
from typing import List

from nltk.tokenize import sent_tokenize
from transformers import HfArgumentParser
from sentence_transformers import SentenceTransformer, util
from pyserini.search import LuceneSearcher, FaissSearcher
import pandas as pd
from tqdm import tqdm
import torch

from evaluate import cos_sim, evaluate
from arguments import RunningArguments

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    parser = HfArgumentParser(RunningArguments)
    args, = parser.parse_args_into_dataclasses()
    print(args)

    test_dataset = load_dataset()
    queries = test_dataset.question.tolist()

    documents = []
    if args.r_type == "sparse":
        documents = run_sparse_retriever(queries, index_path=args.s_index, top_k=5)
    elif args.r_type == "dense":
        documents = run_dense_retriever(queries, dense_index=args.d_index, sparse_index=args.s_index, top_k=5)
    
    predictions = run_reader(queries, documents, args.reader_path, top_k=5)
    evaluate(test_dataset, predictions)

def load_dataset():
    DATASET_PATH = "./datasets/test.csv"
    test_dataset = pd.read_csv(DATASET_PATH)
    return test_dataset

def run_reader(queries: List[str], documents, reader_model: str, top_k: int = 10):
    if reader_model == "baseline":
        reader_model_path = "all-mpnet-base-v2"
    else:
        reader_model_path = os.path.join("./model", reader_model)
    
    model = SentenceTransformer(reader_model_path, device=device)
    model = model.to(device)
    results = []
    q_embeddings = model.encode(queries, convert_to_tensor=True)

    print("Reading Sentences:")
    for i, query in enumerate(tqdm(queries)):
        q_embedding = q_embeddings[i]
        retrieved_documents = documents[i]
        
        predictions = []
        for document in retrieved_documents:
            doc_sentences = document["sentences"]
            sent_embeddings = model.encode(doc_sentences, convert_to_tensor=True)
            scores = util.cos_sim(q_embedding, sent_embeddings).squeeze(0)
            
            for score, sent in zip(scores, doc_sentences):
                predictions.append([document["id"], sent, score])
            
        predictions = sorted(predictions, key=lambda x:x[-1], reverse=True)
        results.append(predictions[:top_k]) # top-10 sentences 

    return results

def run_sparse_retriever(queries: List[str], index_path: str, top_k: int = 10, ) -> List:
    index_path = os.path.join("./indexes", index_path)
    s_searcher = LuceneSearcher(index_path)

    results = []
    for query in queries:
        hits = s_searcher.search(query, k=top_k)

        matched_documents = []
        for i, hit in enumerate(hits):
            json_doc = json.loads(hit.raw)
            json_doc["sentences"] = sent_tokenize(json_doc["contents"])
            matched_documents.append(json_doc)        
        
        results.append(matched_documents)
    return results

def run_dense_retriever(queries: List[str], dense_index:str, sparse_index: str,
                        top_k=10):
    sparse_index = os.path.join("./indexes", sparse_index)
    dense_index = os.path.join("./indexes", dense_index)

    s_searcher = LuceneSearcher(sparse_index)
    d_searcher = FaissSearcher(
        dense_index,
        "facebook/dpr-question_encoder-multiset-base"
    )

    results = []
    for query in queries:
        hits = d_searcher.search(query, k=top_k)

        matched_documents = []
        for i, hit in enumerate(hits):
            doc = s_searcher.doc(hit.docid)

            json_doc = json.loads(doc.raw())
            json_doc["sentences"] = sent_tokenize(json_doc["contents"])
            matched_documents.append(json_doc)

        results.append(matched_documents)
    
    return results

if __name__ == "__main__":
    main()