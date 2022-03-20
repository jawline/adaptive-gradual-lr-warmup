This is a gradual warmup LR scheduler for pytorch that allows for early fixing of the learning rate through a callback function. This can be useful if you start to hit a plateau before warm up is complete.

This work is based on [pytorch-gradual-warmup-lr](https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/README.md). 

### Install

`$ pip install git+https://github.com/jawline/adaptive-gradual-lr-warmup.git`

### Usage

See adaptive_warmup/test.py for usage.
