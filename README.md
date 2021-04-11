# T-TA

This repository is pytorch version of the paper ["Fast and Accurate Deep Bidirectional 
Language Representations for Unsupervised Learning"](https://www.aclweb.org/anthology/2020.acl-main.76/). 


## Introduction

**T-TA**, or **T**ransformer-based **T**ext **A**utoencoder, 
is a new deep bidirectional language model for unsupervised learning tasks.
T-TA learns the straightforward learning objective, *language autoencoding*,
which is to predict all tokens in a sentence at once using only their context.
Unlike "masked language model", T-TA has *self-masking* mechanism
in order to avoid merely copying the input to output.
Unlike BERT (which is for fine-tuning the entire pre-trained model),
T-TA is especially beneficial to obtain contextual embeddings, 
which are fixed representations of each input token
generated from the hidden layers of the trained language model.

T-TA model architecture is based on the [BERT](https://arxiv.org/abs/1810.04805) model architecture,
which is mostly a standard [Transformer](https://arxiv.org/abs/1706.03762) architecture.
This code is based on [huggingface's transformers](https://github.com/huggingface/transformers),
which includes methods for building customized vocabulary, preparing the Wikipedia dataset, etc.

## Usage
```python
from tta.modeling_tta import TTALMModel
from transformers import AutoTokenizer

model = TTALMModel('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
```
* model is not trained