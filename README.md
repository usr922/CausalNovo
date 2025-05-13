# CausalNovo

## Overview

The Pytorch implementation of CausalNovo.

## Environment

```
conda create -n causalnovo python==3.10 
conda activate causalnovo
pip install -r requirements.txt
```




## Data Preparation

Simply follow NovoBench for data preparation.



## Usage



### 1. Training

```bash
bash scripts/train.sh
```



### 2. De Novo Sequencing

```bash
bash scripts/eval.sh
```



### 3. Evaluation

```
python get_result.py
```



