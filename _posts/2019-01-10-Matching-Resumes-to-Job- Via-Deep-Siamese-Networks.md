## Matching Resumes to Jobs via Deep Siamese Network

#### Introduction 

* Given a collection of semi structured job description J and some known matched and unmatched resume / job description pairs *<r,j>* the task is to retrieve matching job descriptions for any existing or new resume *r*.  

* Traditional engines fail to understand the underlying semantic meanings of different resumes and have not kept pace with recent progress in machine learning and natural language processing (NLP) techniques. 

* These solutions are generally driven by manually engineered features and set of rules with predefined weights to keywords which lean to an inefficient and ineffective search experience for job candidates and not generally scalable. 

* Text based tasks rely heavily on representations that could be effectively learned. such representation must capture the underlying semantics among textual elements, where textual elements can be sequences, words or characters.

* State of the art models like Bag of words and TF-IDF are effective in many NLP tasks, in the context of understanding the underlying semantics they are inefficient due to their inherent term specificity.

* Models like Glove, Word2Vec are less effective at document level.

* Siamese Networks possess the capability of learning a similarity metric from the available training records. Here Siamese networks posses the capability of learning a similarity metric from the available training records. here CNN convolutional neural networks with a contrastive loss energy function joining the twin network at the top. The twin network share the weights with eachother 

### Proposed Approach

* Consists of a pair of identical CNN that contains repeating convolution, Max pooling and leaky rectified linear unit layers with a fully connected layer at the top.
* CNN is used because any part of text in resumes and job description can influence the semantics of the word. In order to effectively capture the CNN should see the entire input at once hence CNN is used, Use of CNN could be contrasted by use of LSTM networks. LSTM usually read from left to right hence Bidirectional LSTM are required. which is even more computationally intensive. hence CNN was considered. CNN grows a larger receptive field as we stack more and more layers. It gives a desired effect in a controllable fashion with a low computational cost.
* CNN gives a non linear projection to the resumes and JD in semantic space. The semantic vectors yielded from CNN are connected to a layer measuring similarity between resume and JD. The contrastive loss function combines the measured distance and the label. The gradient of the loss function with respect to weights and biases shared by subnetworks are computed using back propagation.
* Parameter sharing is the highlight of siamese networks. Training the network with a shared set of parameter not only reduce number of parameter but also represent consistency of the representation of JD and Resume in a semantic space. The shared parameters are learned with the aim to minimize / Maximize the distance between resumes and JD.  Doc2Vec is used for document embedding of resume and jd. This vector representation is given as input to the network.

### Experiments

**Baseline Method** - 6 commonly used representations 1) **Word n_grams** 2) **TF-IDF** - Baseline methods to compare with due to their effectiveness in variety of NLP tasks 3) **BOW** : The word frequency from training text is selected and count of each word is used as features. 4) **Bag of Means** : The avearage word2vec embedding of the training data is used as a feature set. 5) **Doc2Vec** : An unsupervised algorithm that learns fixed-length feature representation from variable length pieces of text. 6) **CNN** . Cosine Similarity was the final step with all these baseline models in order to find the similarity of <r,j> pair. The Value of threshold for cosine similarity is determined by grid search on values.

**Dataset and Experimental Setting** - 

Dataset consists of 1314 resumes and a set of 3809 JDs. hence generating a pair of 5,005,026 distinct <r,j> pair with label as 0 or 1. Annotated the corpus in semi supervised fashion. First few were manually annotated by interns. Each resume and JD was preprocessed by *lower casing*, *stemming* and *special character removal*. Performed training in **batches of size 128** . **length of semantic vector was 256**.  **depth of CNN was 4**, **Kernel width of max pooling is 100** and **convolution is 10** and **learning rate was set to 0.01**. **Adam optimization** is used to update parameters of the subnetworks.

**Result and Analysis** 

Table shows the result of the various methods on task of matching resumes to jobs. Compare with 6 baseline methods. Performance of proposed siamese adaptation of CNN is significantly better than a all of the baseline models.







  ​     