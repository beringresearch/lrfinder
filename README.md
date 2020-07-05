# Implementation of the LR Range test

Learning Rate is an important tunable hyperparameter that affects model performance. This repository is a `tf.keras` implementation of the learning rate range test described in [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) by Leslie N. Smith.

## Rationalle

The learning rate range test is a test that provides valuable information about the optimal learning rate. During a pre-training run, the learning rate is increased linearly or exponentially between two boundaries. The low initial learning rate allows the network to start converging and as the learning rate is increased it will eventually be too large and the network will diverge. Typically, a good static learning rate can be found half-way on the descending loss curve.

## Installation

```
git clone https://github.com/beringresearch/lrfinder/
cd lrfinder
python3 -m pip install --editable .
```