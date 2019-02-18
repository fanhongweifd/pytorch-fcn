# pytorch-fcn

[![PyPI Version](https://img.shields.io/pypi/v/torchfcn.svg)](https://pypi.python.org/pypi/torchfcn)
[![Python Versions](https://img.shields.io/pypi/pyversions/torchfcn.svg)](https://pypi.org/project/torchfcn)
[![Build Status](https://travis-ci.org/wkentaro/pytorch-fcn.svg?branch=master)](https://travis-ci.org/wkentaro/pytorch-fcn)

PyTorch implementation of [Fully Convolutional Networks](https://github.com/shelhamer/fcn.berkeleyvision.org).


## Requirements

- [pytorch](https://github.com/pytorch/pytorch) >= 0.2.0
- [torchvision](https://github.com/pytorch/vision) >= 0.1.8
- [fcn](https://github.com/wkentaro/fcn) >= 6.1.5
- [Pillow](https://github.com/python-pillow/Pillow)
- [scipy](https://github.com/scipy/scipy)
- [tqdm](https://github.com/tqdm/tqdm)


## Installation

```bash
git clone https://github.com/fanhongweifd/pytorch-fcn.git
git checkout transport
python setup.py install
```

## Training

See [train example](examples/voc/train.sh).

Input file type should be xlsx or pickle (pickle perfer).
Use script [transfer](torchfcn/datasets/transport.py) to transfer xlsx to pickle


## Model
Building your own model in [model example](torchfcn/models/fcn8s_pm25.py)


## Log parameter
lr: learn rate
smape: Symmetric mean absolute percentage error
<img src=".readme/smape.svg" width="20%" />  
mce_loss: Mean squared loss
predict_array mask: predict score
target_array mask: ground truth label