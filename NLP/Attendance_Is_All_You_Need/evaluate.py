from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import pandas as pd

def evaluate(test_dataset: pd.DataFrame, predictions):
    # for idx, data in test_dataset.iterrows():
    #     prediction = predictions[idx]
    #     print(f"Result for Query: {data.question} in {data.chapter} (Page {data.page}):")
    #     print(f"Ground Truth: {data.answer}")
    #     print("-----------------")

    #     for sent in prediction:
    #         print(sent)                
        
    #     print("=============================")
    
    metric = cos_sim(predictions, test_dataset.answer.tolist())

    print(f"Score: {metric}")

def cos_sim(predictions, labels, threshold=0.8):
    '''
        predictions: N x k sentences
        labels: N sentences
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_name = "bert-large-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)

    N = len(labels)
    k = len(predictions[1])
    sim_fn = nn.CosineSimilarity(dim=0)

    acc = 0.0
    sum = 0.0
    
    for i in tqdm(range(N)):
        encoded_input = tokenizer(labels[i], return_tensors='pt').to(device)
        label_embedding = model(**encoded_input)[1].flatten()

        for sent in predictions[i]:
            text = sent[1]
            encoded_input = tokenizer(text, return_tensors='pt').to(device)
            embedding = model(**encoded_input)[1].flatten()

            score = sim_fn(label_embedding, embedding)
            sum += score.detach().cpu().sum()
            if score > threshold:
                acc += 1.0

    return sum / float(N * k)
