## Learning Text Similarity with Siamese Recurrent Networks

Authors - Paul Neculoiu, Maarten Versteegh and Mihai Rotaru. 

Link to the Paper  - http://www.aclweb.org/anthology/W16-16#page=162

#### Abstract 

* Presents a deep architecture for learning similarity metric on variable length character sequences.
* Model combines a stack of character-level bidirectional LSTM's with a Siamese architecture.
* Learns to project variable length strings into a fixed dimensional embedding space by using only information about similarity between pairs of strings.
* Model could be applied to task of job title normalization based on a manually annotated taxonomy. model learns a representation that is selective to differences in the input reflect semantic differences (eg:- Java developer vs HR manager) but also invariant to non semantic string differences (eg- Java developer vs Java programmer)

#### Introduction

* Representation that express semantic similarity and dissimilarity between textual elements is important in NLP.
  * Word similarity models (Mikolov 2013) is applied in diverse settings such as sentiment analysis and recommender system.
  * Semantic textual similarity is used in automatic summarization (Ponzanelli 2015), debate analysis (Boltuzic and Snajder 2015) and paraphrase detection (Socher 2011).
* Measuring semantic similarity between texts is necessary for Information Extraction (IE). This process is also called Normalization ie. to put pieces of information in standard format. eg:- '12pm, noon, 12:00h' all should map to same representation. Normalization is essential for retrieving actionable information from free unstructured text.
* This paper presents a system for job title normalization. task here is to receive an input string and map it to one of a finite set of job codes which were predefined externally. It could be considered as a highly Multi-class classification problem, but here the focus is on learning representation of the strings such that synonymous job titles are close together.

#### Related Work

* Representation learning through neural network (Auto encoders, Hinton 2006)  is invariant to differences in the input that do not matter for that task and selective to differences. 
* Siamese network (Bromley 1993) is an architecture for non linear metric learning with similarity information. The network naturally learns representations that reveals the invariance and selectivity through explicit information about similarity between pairs of object. Siamese network learns an invariant and selective representation directly through the use of similarity and dissimilarity information.
* An Autoencoder learns invariance through added noise and dimensionality reduction in the bottleneck layer and selectivity through the condition that the input should be reproduced by the decoding part of the network.
* Siamese architecture was originally applied to signature verification (Bromley 1993) and has been used in Vision application. Siamese convolutional networks are used for face verification (Chopra 2005) and dimensionality reduction on image features (Hadsell 2006). They have been also used for diverse tasks as unsupervised acoustic modelling (Synnaeve 2014), Learning food preference (Yang 2015), Scene detection (Baraldi, 2015). In NLP Siamese network with convolutional layers have been applied to matching sentences (Hu 2014) and for learning semantic entailment (Mueller 2016).
* Job Title Normalization is framed as a classification task (Javed 2014)  as it has large number of classes. Multi stage classifier have shown good results. But there are disadvantages to this approach
  * Expense of data acquisition for training number of classes will exponentially increase the data requirement. 
  * Once a classification error is identified, we need to retrain the entire classifier with new sample added to correct the class 
  * Using traditional classifier do not allow for transfer learning.
* Use of String similarity measure to classify input strings here the advantage is there is no need to train the system hence improvement can be made by adding job title strings to the data.
* Modeling similarity directly based on pairs of inputs Siamese networks lend themselves well to the semantic invariance present in job title normalization.

#### Siamese Recurrent Neural Network

* Recurrent Neural Networks are neural networks adapted for sequence data. LSTM variant of RNN has particularly had success in tasks related to natural language processing like text classification and language translation.  

* Bi directional RNN incorporate both future and past context by running the reverse f input through a separate RNN. The output of combined model at each time step is simply the concatenation of outputs from the forward and backward networks. 
* Siamese networks are dual branch networks with tied weights, they consist of the same network copied and merged with an energy function. The training set for Siamese network consists of triplets (x1,x2 and y) where x1 and x2 are character sequences and y indicates whether x1 and x2 are similar or dissimilar (0 or 1). The aim of training is to minimize the distance in an embedding space between similar pairs and maximize the distance between dissimilar pairs.

#### Contrastive Loss Function

* The Network contains 4 layers of Bidirectional LSTM nodes. the activations at each time step of the final BLSTM are averaged to produce a fixed dimensional output. This output is projected through a single densely connected feed forward layer.  Energy of the model E is the cosine similarity between embeddings of x1 and x2. 
* The Network used has 4 BLSTM layers with 64 dimensional Hidden vectors h and memory c. there are connections at each time step between layers. Outputs of the last layer are averaged over time and this 128 dimensional vector is used as input to a dense feed forward layer. 
* The input strings are padded to produce a sequence of 100 characters with input string randomly placed in this sequence. The parameters of the model are optimized using Adam method and trained until convergence. Dropout technique (Srivastava 2014) on recurrent units and between layers (probability .4) is done to prevent overfitting.

#### Experiment 

* Small dataset based on handmade taxonomy of job titles is used. After each experiment the dataset is augmented by adding new source of variance. 
* **Baseline** - ngram matcher (Daelemans 2004). given an input string, this matcher looks up closest neighbour from base taxonomy by maximizing a similarity scoring function. The matcher subsequently labels input string with neighbour's group label.  This similarity function has the properties that it is easy to compute and doesn't require any learning and is particularly insensitive to appending extra words in the input string.
* Here the test set consists of pairs of strings, the first of which is the input string and the second a target group label from the base taxonomy. The network model projects the input string into the embedding space and searched for its nearest neighbor under cosine distance from the base taxonomy. The test records a hit only if the neighbors group label matches the target.

#### Data and Data Augmentation 

* Handmade Job Title taxonomy partition set of 19,927 job titles into 4,431 groups. Job Titles were manually and semi automatically collected from resumes and vacancy postings. Each was manually assigned a group such that job titles in a group are close together.

* The wide variety of different semantic relations between and within groups is considered as an asset to be exploited by the model. 

* The groups are not equal in size largest group has 130 job titles and smallest group has just one. The long tail may affect the models ability to accurately learn to represent smallest group. 

* 4 stage process starting from base taxonomy of job titles at each stage an augmentation of data which focuses on a particular property and a test that probes the model for behavior related to that property. each stage build on next stage hence previous augmentations are included.

* The dataset consists of pairs of strings sampled from the taxonomy in a 4:1 ratio of between class (negative pairs) to within-class (positive pairs)  

  **Testing Criteria**

* *Typo and Spelling invariance* - Slight differences in spelling from taxonomy should be intelligently picked. to induce this invariance we augment the base taxonomy by extending it with positive sample pairs consisting of job title strings and the same string but with 20% characters randomly substituted and 5% deleted. 

* *Synonyms* - Model should be invariant to synonym substitution we augment the data set by substituting words in job titles by synonyms. first source is a manually constructed job title synonym set consisting of around 1100 job titles each with between 1 and 10 synonyms. Second source is by induction the complements of the matching strings from a synonym candidate. 

* *Extra Words*  - To be useful  model should be invariant to the present of additional words. 

* *Feedback* - Model should be corrigible. we should be able to append the model when it performs badly.

#### Result

Comparison was done with baseline n-gram system and proposed neural network models on the four tests. Both n-gram matching system and the proposed model have near complete invariance to the simple typos.

#### Discussions

* A model architecture for learning text similarity based on Siamese recurrent neural networks is presented. This architecture learns a series of embedding spaces based on specific augmentation of the dataset used to train the model. The embedding spaces captured important invariances of the input, the model was invariant to spelling variations, synonym replacements and superfluous words. 
* The ability of the system to learn invariances are due the contrastive loss function combined with the stack of recurrent layers. Using separate loss functions for similar and dissimilar samples helps the model maintain selectivity while learning invariances over different sources of variability.

#### Further Improvements 

* Incorporating convolutional layers in addition to the recurrent layers.
* Investigating a triplet loss function instead of contrastive loss used in this study
* Comparison to a stronger baseline would serve further development.
* Job Title taxonomy used in the current study exhibits a hierarchical structure that is not fully exploited. further research could attempt to learn a single embedding which would preserve the separation between groups at different levels of hierarchy.

















