## pImplementing RNN / LSTM in Tensorflow

So i started learning about LSTM and RNN through Christopher Olah's blog about the same. Please find it here - http://colah.github.io/posts/2015-08-Understanding-LSTMs . It gives a great intuitive understanding about the topic that I'm trying to implement.

so lets start with some dummy data - 

* Input Data - At time step t, X_t  has a 50% chance of being 1 (and a 50% chance of being 0). E.g., X might be [1, 0, 0, 1, 1, 1 … ].
* Output Data - At time step t, Y_t  has a base 50% chance of being 1 (and a 50% base chance to be 0). The chance of Y_t  being 1 is increased by 50% (i.e., to 100%) if X_t - 3 is 1, and decreased by 25% (i.e., to 25%) if X_t - 8 is 1. If both X_t - 3 and X_t − 8 are 1, the chance of Y_t being 1 is 50% + 50% - 25% = 75%.

Lets generate this data in python

```python
import numpy as np
import tensorflow as tf
%matplotlib inline
import matplotlib.pyplot as plt

def gen_data(size=1000000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)

def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)

def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)
```






