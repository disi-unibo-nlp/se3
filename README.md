# Semantic Self-Segmentation for Abstractive Summarization of Long Documents in Low-Resource Regimes

## Introduction
The quadratic memory complexity of transformers prevents long document summarization in low computational resource scenarios. State-of-the-art models need to apply input truncation, thus discarding and ignoring potential summary-relevant contents, leading to a performance drop. Furthermore, this loss is generally destructive for semantic text analytics in high-impact domains such as the legal one. In this paper, we propose a novel semantic self-segmentation (Se3) approach for long document summarization to address the critical problems of low-resource regimes, namely to process inputs longer than the GPU memory capacity and produce accurate summaries despite the availability of only a few dozens of training instances. Se3 segments a long input into semantically coherent chunks, allowing transformers to summarize very long documents without truncation by summarizing each chunk and concatenating the results. Experimental outcomes show the approach significantly improves the performance of abstractive summarization transformers, even with just a dozen of labeled data, achieving new state-of-the-art results on two legal datasets of different domains and contents. Finally, we report ablation studies to evaluate each contribution of the components of our method to the performance gain.

## Setup
You will need to install several packages:
```
pip install -r requirements.txt
```

## Deep Metric Learning
The first step is to perform the metric learning training:
```
batch = 8
epochs = 1
model = "nlpaueb/legal-bert-base-uncased"
seed = 1234

python metric_learning.py \
  --batch $batch \
  --epochs $epochs \
  --model $model \
  --seed $seed
```

## Semantic Self-Segmentation
The second step is to segment the documents:
```
checkpoint = "allenai/led-base-16384" # "facebook/bart-base" for BART
dataset = "billsum or austlii"
dataset_only = "" # If you have just one dataset to segment, fill this field
min_input_len = 1024 # 512 for BART
max_input_len = 2048 # 1024 for BART
max_output_len = 1024 # 512 for BART
model = "models/metric_learning_nlpaueb/legal-bert-base-uncased"
no_save = False # True if you do not want to save results (for toy experiments)
seed = 1234
toy = 0 # i > 0 to process just i documents

python autofocus/segmentation.py \
  --checkpoint $checkpoint \
  --dataset $dataset \
  --dataset_only $dataset_only \
  --min_input_len $min_input_len \ 
  --max_input_len $max_input_len \
  --max_output_len $max_output_len \
  --model $model \
  --no_save $no_save \
  --seed $seed \
  --toy $toy 
```

## Abstractive Summarization
The last step is to perform the downstream task fine-tuning on the chunked documents:
```
batch = 1
batch_eval = 2
checkpoint = "allenai/led-base-16384" # "facebook/bart-base" for BART
dataset = "billsum or austlii"
epochs = 5
full = False # To use the whole documents (not chunked)
grad_acc = 1
max_input_len = 2048 # 1024 for BART
max_output_len = 1024 # 512 for BART
model = "led" # "bart" for BART
predict = False # To just predict
seed = 1234
toy = 0 # i > 0 to process just i documents

python train.py \
  --batch $batch \
  --batch_eval $batch_eval \
  --checkpoint $checkpoint \
  --dataset $dataset \
  --epochs $epochs \
  --full $full \
  --grad_acc $grad_acc \  
  --max_input_len $max_input_len \
  --max_output_len $max_output_len \
  --model $model \
  --predict $predict \
  --seed $seed \
  --toy $toy 
```
