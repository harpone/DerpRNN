# DerpRNN
Deep RNN models with Theano.


**Why 'DerpRNN'? Because 'DeepRNN' was already taken!**

This is a smallish Python module for deep recurrent neural networks using **Theano**. Evolving through time and the layers is performed all in a nested theano.scan loop, so the implementation is in this way a bit different than in e.g. Keras. Here's a list of some of the features (of the `DeepRNN` class):

- Hidden recurrent layers can be SRU (standard recurrent layer) or GRU (all same type and shape!)
- `tanh` "read in" layer before the recurrent layers
- Readout layer can be `RBM`, `tanh`, `sigmoid` or `softmax`
- There's also a fully translation invariant version (`InvariantDeepRNN`, but it's not quite done yet...)

Requirements are the usual scientific python ones, plus of course **Theano**. Also `python-midi` is needed for processing the midi data. You may also need need **cython**.

There's a seup script, so you *should* be able to install the module and the dependencies by


If that fails, you can do `` and then `python setup.py build_ext --inplace` to compile the cython modules (although you may not need to do that at all, depending on your machine).


![](https://github.com/harpone/DerpRNN/blob/master/Mr-derp.png "Mr. Derp")

