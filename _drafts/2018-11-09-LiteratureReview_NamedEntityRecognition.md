### Literature Review of Named Entity Recognition using Neural Networks 

This is a short literature review of the detailed study conducted by Fabian Karl on the topic of Named Entity Recognition, the complete paper is available in this link :  [Independent Study by Fabian Karl (University of Hamburger)](https://www.inf.uni-hamburg.de/en/inst/ab/lt/teaching/independent-studies/201711-karl-ner-independent.pdf).

**Named Entity Recognition** - in simple words it is the task of finding or recognising named entities from a text / corpus. Named Entity could be anything from a person, organisation, Location or anything of that sort which are of interest to the user. The goal of the study is to replicate the state of the art  performance in NER without any handcrafted rules but only using pre labelled training data.

**Dataset Used** - two labelled data collection were used for training and testing

* CoNLL-2003 - English and German
* GermEval-2014 Dataset

**Neural Network** 

* Recurrent Neural Network - RNN's get a sequence of features as input where each feature could stand for one time step of some process, However RNN has a serious flaw of Exploding / vanishing gradient. because of which we have considered LSTM cells which are modified RNN. Major characteristic of LSTM cell is the various kind of forget gates. LSTM are widely used in Named Entity recognition.
* Convolutional Neural Network - CNN connects only a certain number of  nodes to one of its preceding nodes where in a regular feed forward neural network connects to a certain number of nodes to one of its preceding nodes.  CNN's are state of art network for image recognition tasks.
* Conditional Random Fields - An additional conditional random field CRF instead of last softmax layer of one of the previously  described  neural networks allows to take the context of the output tags into account. Hence it can make predictions about the out coming tags on the basis of the whole output sequence.

**Methods**

Creating a numeric representation of textual input is essential in NLP for embedding it into a vector representation. 

* **One-Hot** encoding is the best and easiest way to encode everything that is present. However this method has its own drawbacks, results in n vectors of size n (number of words) and has no meaningful representation

* Embed each word based its context First they are one Hot encoded and trained on on their context of words  with one hidden layer. This allows for a meaningful comparison between vectors. this method allows to embed vectors into smaller vectors eg- 300 size vectors.

* **Fasttext Model** -  it is a pretrained model that uses continuous word representations instead of discrete representations of every word. Here a a vector representation is associated to each charater n grams with words being the sum of these representation.

* **Word2vec** and **BPEmb** - They performed worse than Fasttext model

**Bi Directional Context** 

In order to take context information into account when predicting  the tag of a word, the words before and after the word in question were also embedded and used as an additional  information. The sentence was split into two lists where the first list contained all the embedding of word from the beginning of the sentence until the word in question inclusive, second list contains all the embeddings from the last word to the word in question. 

**Model Architecture** 

A modified Bi LSTM neural-net was used. After embedding  2 LSTM networks with 300 cells were added to the model, they were concatenated and result was fed to into 3 connected dense layer with 300 , 100 and 9 units respectively with a dropout of 0.6. If the CRF layer was not used, the Softmax activation function was applied for the last layer. ReLU activation was used for all the other layers. If CRF function was used in the last layer then ReLU was used instead of softmax. The last layer was then reshaped into a sequence length of 1 and given into CRF layer which returned a one hot encoding over the nine  output categories. The architecture resulted in having around 1.7 million trainable parameters. Batch Size was set to 512  for all the experiments  without CRF and 300 for all others.

**Character Embedding**

The word is converted into characters and the ASCII representation is saved into a List. This List is reversed and used as separate input into the dense layer and then concatenated  with the first drop-out layer after concatenation of the two LSTM layers.

**CNN instead of LSTM** 

This was the second architecture they had experimented with. Two LSTM layers were replaced by number of convolutional layers spanning over different number of words. Each convolutional layer consists of fixed number of convolutional filters, each spanning over a matrix of 300*n . The idea is to use multiple smaller filters in order to  capture spacial and local information.

**Other Related Works**

Here the aim of the study was to replicate the state of the art behaviour for the NER task. Hence elements of the neural network architecture of other proven studies were taken into consideration. Use of Bi LSTM with CRF layer and character embedding had a performance score of 90.94 on english and 78.76 on german which were significant improvement at those current standards.

**Discussion**

* **Character Embedding** - the results showed that when using a char representation of the word in question next to the bidirectional embedding of the whole sequence with a pretrained fasttext word embedding model gives the best performance. The fasttext embedding has a slight disadvantage that it converts all the Upper case letters to lowercase before reading it.

* **Conditional Random Field** - CRF conditions showed inferior performance than softmax layer. a good explanation couldnot be found for this. Hyper parameter tuning was not systemic that could be a valid reason why it didn't perform well.

**Conclusion**

This proves the capabilities of neural architectures to learn tagging task (and categorisation tasks) with no external knowledge needed. It also shows that the importance and performance  of neural network architectures for  natural language tasks will further increase in the future




