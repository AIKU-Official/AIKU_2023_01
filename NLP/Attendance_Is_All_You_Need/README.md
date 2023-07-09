Member : **황상민**, 김예랑, 김민성, 김성찬

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