Deep Info Max
=============

Reimplementation of [Learning deep representations by mutual information
estimation and maximization](https://arxiv.org/abs/1808.06670)

**Abstract:**

*R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Phil Bachman, Adam Trischler, Yoshua Bengio*

> In this work, we perform unsupervised learning of representations by maximizing mutual information between an input and the output of a deep neural network encoder. Importantly, we show that structure matters: incorporating knowledge about locality of the input to the objective can greatly influence a representation's suitability for downstream tasks. We further control characteristics of the representation by matching to a prior distribution adversarially. Our method, which we call Deep InfoMax (DIM), outperforms a number of popular unsupervised learning methods and competes with fully-supervised learning on several classification tasks. DIM opens new avenues for unsupervised learning of representations and is an important step towards flexible formulations of representation-learning objectives for specific end-goals.

Dataset: CIFAR10
Encode & Dot Architecture

Table 1. (from paper)

| Model      |  conv |   fc   |  Y    | alpha | beta | gamma |
|:---        |  :--- |  :---  |:---   | :---  | :--- |:---   |
|DIM(G)      |  52.20| 52.84  | 43.17 | 1     | 0    | 1     |
|DIM(L) (JSD)|  73.25| 73.62  | 66.96 | 0     | 1    | 0.1   |


To train the global encoder DIM(G):
```bash
python trainer.py --alpha=1 --beta=0 --gamma=1 --epochs=40
```
Should achieve 43% classification accuracy (replicating column Y in table 1.)

To train the local encoder DIM(L) (JDS):
```bash
python trainer.py --alpha=0 --beta=1 --gamma=0.1 --epochs=40
```
Currently doesn't match results from table 1.
