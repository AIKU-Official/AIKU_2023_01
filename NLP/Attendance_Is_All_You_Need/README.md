# Attendance is All You Need
Member : **황상민**, 김예랑, 김민성, 김성찬

## 주제 선정 배경

![motivation](https://github.com/AIKU-Official/AIKU_2023_01/assets/29402508/c053eff5-75f6-4355-a6fa-c8b0add4c06c)

수업을 듣다 보면 강의 노트가 교과서 기반으로 만들어지기는 하지만, 강의 노트의 내용과 수업에서 구두로만 설명하신 내용을 교과서에서 바로 찾기가 어려운 경우가 있습니다. 그럴 때 교과서에서 원하는 내용을 검색해보곤 하는데, **단순 키워드 검색**으로는 찾고자 하는 내용과 관련된 내용을 찾아주지 못할 때가 많습니다.

 따라서 저희는 **의미 정보를 기반**으로 교과서에서 유사한 내용을 찾아 주는 프로젝트를 기획하게 되었습니다. 

## 프로젝트 요약

NLP 모델을 활용해서 교과서 파일과 query를 입력으로 주었을 때 해당 query과 가장 연관이 있는, 즉 query와 가장 유사도가 높은 문장과 그 문장이 포함된 페이지를 찾아주는 프로젝트입니다. 

## Dataset

**1) STS benchmark**

Sentence Similarity task 학습에 사용되는 영어 데이터셋입니다.

[STSbenchmark - stswiki](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark)

**2) Custom Test Dataset**

총 8개의 Chapter로 구성된 James.W.Kurose <Computer Networking : A top-down approach> 의 각 챕터 마지막 부분에 있는 Questions의 양식을 참고하여, 각 챕터의 소단원별로 4개씩 질문을 생성하였습니다. 챕터(소단원), 질문의 출처가 되는 페이지, 질문, 페이지 내에서 질문에 대한 답변이 되는(혹은 가장 관련이 있는) 문장 네 가지로 이루어진 csv 데이터셋을 통해 태스크를 진행하였습니다.

<Computer Networking : A top-down approach>의 챕터 구성은 A.B.C 형태로, 대단원>소단원>소주제 형태로 총 3분류의 구성입니다. 다만, 내용의 다양성 및 데이터셋 생성의 편리성을 고려하여, A.B (대단원>소단원) 분류까지만 고려하여 데이터셋을 생성하였습니다. 데이터셋 생성 과정에서는, 기존 책의 Questions와 비슷한 형태로 질문을 구성하였으며, 각 단원의 핵심 키워드가 포함될 수 있도록 생성하였습니다.

**Dataset preview**

![dataset_preivew](https://github.com/AIKU-Official/AIKU_2023_01/assets/29402508/06365ecb-83f3-45d0-bb89-e330a84a723c)
## Methodology

**1) Sentence Similarity** 

Query와 교과서 내 문장 사이의 Cosine Similarity를 계산해 두 문장이 내용적인 측면에서 얼마나 유사한지를 판단합니다. 이를 위해 pre-trained된 모델로 두 문장의 sentence에 대한 embedding vector를 생성합니다. Pre-trained 모델을 Contrastive Learning과 TSDAE 학습 방식을 통해 Custom Dataset으로 Fine-tuning을 진행했습니다.

**2) Retrieval-reader Framework**

Open domain Question Answering에서 사용하는 retrieval-reader framework를 차용하여, 여러 document 중에서 한 document에 대해 주어진 질문에 대해 답을 찾는 방법을 시도했습니다. document는 교과서를 페이지 단위로 나눈 것으로 하였으며, query와 연관성 높은 document를 찾기 위해 여러 모델을 통해 시도해 보았습니다. 시도한 방법에는 TF-IDF/BM25, Dense retrieval model 등이 있습니다. 

**3) Overall Framework**

교과서 파일을 모델이 처리할 수 있는 형태로 전처리한 뒤 해당 데이터에서 Retrieval-reader Framework를 통해 Query와 가장 연관 있는 k개의 페이지를 찾습니다. 이후 Fine-tuned 모델에 다시 query와 k개의 페이지를 input으로 넣어주어 k개의 페이지에서 Query와 가장 연관있는 문장을 output으로 출력합니다.

![framework](https://github.com/AIKU-Official/AIKU_2023_01/assets/29402508/521efc29-96e1-40d0-9737-63b06e0f4379)

# Usage

## Building index:

### Building Sparse index

```bash
python -m pyserini.index.lucene --collection JsonCollection --input ./resources/target --index ./indexes/indexes_page_filtered --generator DefaultLuceneDocumentGenerator --threads 1 --storePositions --storeDocvectors --storeRaw
```

### Building Dense index

```bash
python -m pyserini.encode input --corpus ./resources/document_dpr.json --fields text --delimiter "\n" --shard-id 0 --shard-num 1 output  --embeddings ./indexes/indexes_dpr --to-faiss encoder --encoder facebook/dpr-ctx_encoder-multiset-base --fields text --batch 32  --fp16
```


Pyserini expects the fields in contents are separated by `\n`(delimiter parameter)

In this case, we use `facebook/dpr-ctx_encoder-multiset-base` model for context encoder. (`facebook/dpr-question_encoder-multiset-base)` for query encoder in reader)

## Train

```bash
python train.py
```

Path variables in code should be modified according to your environment.

## Inference

### With sparse retrieval

```bash
python run.py --r_type sparse --s_index {sparse index folder name} --reader {model folder name}
```

One example can be:

```bash
python run.py --r_type sparse --s_index indexes_page_filtered --reader finetuned-30-70
```

### With dense retrieval

```bash
python run.py --r_type dense --s_index {sparse index folder name} --d_index {dense index folder name} --reader {model folder name}
```

In our project, dense retrieval also requires sparse indexes in order for particular document information.

One example can be:

```bash
python run.py --r_type dense --s_index indexes_page_filtered --d_index indexes_dpr --reader finetuned-30-70
```
