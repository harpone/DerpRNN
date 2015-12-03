# DerpRNN
Deep RNN models with Theano.


**Why 'DerpRNN'? Because 'DeepRNN' was already taken!**

This is a smallish Python module for deep recurrent neural networks. Evolving through time and the layers is performed all in a nested theano.scan loop, so the implementation is in this way a bit different than in e.g. Keras. Here's a list of some of the features:

- Hidden recurrent layers can be SRU (standard recurrent layer) or GRU (all same type and shape!)
- `tanh` "read in" layer before the recurrent layers
- Readout layer can be `RBM`, `tanh`, `sigmoid` or `softmax`


![](https://github.com/harpone/DerpRNN/blob/master/Mr-derp.png "Mr. Derp")

