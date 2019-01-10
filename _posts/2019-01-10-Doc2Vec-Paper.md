### Distributed Representations of Sentences and Documents



Most algorithms typically require text input to be represented as a fixed-length vector. Bag of words / bag of n grams are the most common due to its simplicity, efficiency and often surprising accuracy. Bag of words has many disadvantages, Word order is gone and hence different sentences can have same representation. Bag of n grams considers word order in short context, as it has data sparsity and high dimensionality.

Paragraph vectors is an unsupervised framework that learns continuous distributed vector representations for pieces of texts. The text can be of variable length ranging from sentences to documents.

The vector representation is trained to be useful for predicting words in a paragraph. paragraph vector is concatenated with several word vectors from a paragraph and predicts following word in given context. Both word vector and paragraph vectors are trained by stochastic gradient descent with back propagation. paragraph vectors are unique among paragraph while word vectors are shared. At prediction time paragraph vectors are inferred by fixing word vectors and training the new paragraph vector until convergence.

Paragraph vector is capable of constructing representations of input sequences of variable length. It is applicable for text of any length from sentences, paragraphs to documents.

**Algorithm**

Word vectors are the inspiration for paragraph vectors. Every word is mapped to a unique vector represented by a column in a matrix W.  The column is indexed by position of the word in vocabulary. The concatenation or sum of the vectors is then used as a feature for prediction of the next word in a sentence.  Given a sequence of training words w1,w2,w3,.. wt. Objective of the word vector model is to maximize the average log probability. Prediction is done via a multiclass classifier such as softmax. Neural Network based word vectors are usually trained using stochastic gradient descent and back propagation. Once the training is converged, words with similar meaning are mapped to a similar position in the vector space.

**Paragraph Vectors** - are very much similar to word vectors. Word vectors are asked to contribute to the prediction task about the next word in the sentence. Hence even though word vectors are initialized randomly they eventually capture semantics as an indirect result of prediction task.  the paragraph vector and word vector are averaged or concatenated to predict the next word in a context. The only change when compared to word vector framework is  where h is constructed from W and D. This model is called Distributed Memory model of Paragraph Vectors (PV-DM) as the paragraph vector remembers what is missing from the current context of the paragraph.

The contexts are fixed- length and sampled from a sliding window over the paragraph. The paragraph vector is shared across all the contexts but not with all paragraphs. But the word vector matrix is shared across all the paragraphs. 





