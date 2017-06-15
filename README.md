# Language processing II exam

An exam for a course on language processing

## Requirements
To reproduce the results you need to install python3 and the following
pip3 packages (I highly recommend using a virtualenv):

* Numpy
* Sklearn
* Tensorflow
* Keras

## Repository structure
The preprocessing is done in the ``preprocessing.py`` file, and the results
(included) can be found in the ``data.npz`` file. The proprocessor uses the
data from the ``scaledata`` folder.

Four models are constructed using the proprocessed data: a linear model,
a linear model with a tf-idf score, a recurrent neural network with LSTMs and
a LSTM with tf-idf. They are located in the correspondingly named python files.
Simply running those files with ``python3`` should run the models, given
the prerequisites have been correctly installed.

## Contact
xtp778@sc.ku.dk
