## Siamese Recurrent Architectures for Learning Sentence Similarity

Paper by Jonas Mueller & Aditya Thyagarajan  - http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf

#### Introduction 

Mikolov demonstrated the effectiveness of neural word representations for analogies and other NLP tasks (2013), ever since interest shifted towards extensions of these ideas beyond the individual word-level to larger bodies of text like sentences where a mapping is learned to represent each sentence as a fixed length vector (Kiros 2015 and Mikolov 2014). RNNs are naturally suited for variable length inputs like sentences, LSTM is the most preferred type of RNNs. RNN's are Turing complete optimization of the weight matrices is difficult due to the backpropagated vanishing gradients hence LSTM was preferred. 

Here we consider a supervised learning setting where each training example consists of a pair of sequences of fixed-size vectors along a single label y for the pair. sequences may be of different lengths. The task here can be considered as scoring similarity between sentences given example pairs whose semantic similarity labelled .

#### Related Work



