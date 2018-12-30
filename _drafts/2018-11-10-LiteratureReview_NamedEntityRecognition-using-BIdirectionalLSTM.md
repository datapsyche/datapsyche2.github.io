 

###  Named Entity Recognition with Bidirectional LSTM-CNNs by Jason P.C. Chiu and Eric Nichols.

Reference for the Paper -  [Named Entity Recognition with Bidirectional LSTM - CNN by Jason P.C.Chiu and Eric Nichols](https://arxiv.org/abs/1511.08308).

Github Reference - 





**Introduction** - Named entity recognition (NER) is an important part in NLP and has been dominated by applying CRF, SVM or Perceptron models to hand crafted features. 

* Collobert et al. (2011b) - proposed idea of word embedding trained on large quantity of unlabelled text data with an effective neural network and a little feature engineering. However this doesn't work well with rare words or words that are poorly embedded, hence a new powerful neural network model is needed.
  * RNN's  (1996  - Goller and Kuchler) are popular since 2013 for machine translation and language modelling. 
  * LSTM unit with forget gate improves long term dependencies to be easily learned (Gers et al - 2000). 
  * For Labelling task like NER  and speech recognition a bi directional LSTM  model can take into account infinte amount of context that applies  to any feed -forward model.
  * CNN was investigated for modelling character level information among other NLP tasks.
* hence, a hybrid model involving CNN and bi directional LSTM that learns both character level and word level features presenting the first evaluation of such an architecture on well established english language evaluation datasets.
* Lexicons are considered important for NER performance, hence a new lexicon encoding scheme and matching algorithm that can make use of partial matches and compare it to simpler approach of Collobert.

**Model** 

* In Collobert's model, lookup tables transforms discrete features such as words or characters into continous vector representation, which are then concatenated and fed into a neural network. Here instead of feed forward network, a Bidirectional LSTM network is used. To induce character level features a convolutional neural network is used.

* **Sequence Labelling with LSTM** - A stacked bi-directional  recurrent neural network with long short term memory units are used to transform word features into named entity tag scores. The extracted features of the word are fed into a forward LSTM network and a backward LSTM network. The output of each network at each time step is then decoded by a linear layer and a log softmax layer  into log probabilities for each tag category. These two vectors are simply added to get the final output. This architecture was found to be the best effective one after minor experiments.

* **Extracting Character Features using CNN**  - For each word a convolution and a max layer to extract a new feature vectors such as character embedding and character type. Words are padded with number of special padding characters on both sides depending on the window size of CNN 

* **Word Embedding** - the best model used the publicly  available 50 dimensional word embedding released by Collobert which were trained on Wikipedia and Reuters Dataset that are publicly available. all words were lower cased before passing through the lookup table to convert the word embedding. the pretrained embedding was allowed to be modified during training.

* **Character Embedding** -  We randomly initialized a lookup table with values drawn from a uniform distribution with a range [-0.5 , 0.5] to output a character embedding of 25 dimensions. The character set includes all unique characters in CoNLL-dataset plus the special token paddings and unknown.

* **Capitalization** - Collobert's method of using separate lookup table to add for a capitalization feature with options, All Caps, Upper Initial, lowercase, mixed case and no info.

* **Lexicons** - lexicons were categorised into 4 categories ('Person','Organisation','Location','Misc'). For each lexicon category, it matched every n-gram against entries in the lexicon, A match is successful when the ngram matches the prefix or suffix of an entry and is atleast half the length of the entry. For all categories except Person, partial matches less than 2 token in length were discarded. when multiple overlaps are there maximum matched category is considered. matches are case insensitive. Each category is BIOES (begin, inside, outside, end, single) encoded. The best model used SENNA Lexicon (by Collobert et al) with exact matching and DBpedia lexicon with partial matching with BIOES annotation in both cases.

* **Character Level features** - A lookup table was used to output a 4 dimensional vector representing the type of the character.



**Training and Inference** 
  Implemented using torch7 library (LUA). Training and inference were done on a per sentence level. Initial states of the LSTM are zero vectors. Except for the character and word embeddings whose initialisation were described previously. all lookup tables were also randomly initialised.

**Objective function and Inference** 

  [Requesting to go through the paper as i couldn't comprehend and write here in the blog post]

**Tagging Scheme** - BIOES tagging scheme was used. 

**Learning Algorithm** - Training done by minibatch SGD with a fixed learning rate. Each batch consists of multiple sentences with same number of tokens. Applied dropout to the output nodes of each LSTM layer to reduce overfitting. Other optimization algorithms like momentum, ada delta, RMSProp, were considered but SGD appears to perform well.

**Evaluation** - Evaluation was performed CoNLL-2003 NER shared dataset.

**Dataset Preprocessing** 

  * All digit sequence are replaced by single 0.
  * Before training, Sentences were grouped by word length into mini batches and shuffled.

**Dataset** -  

* CoNLL-2003 dataset consists of newswire from Reuters corpus tagged with four types of named entities : location, organization, person and miscellaneous. Dataset is small hence model has been trained on both training and development sets after performing hyper parameter optimization.
* OntoNotes Dataset - Pradhan et al compiled a core portion of OntoNotes dataset for the CoNLL-2012 shared task and described a standard train/test/dev split which was used for evaluation. This dataset is much larger and has text from variety of sources.

**Hyper parameter Optimisation**  

* Two rounds of hyper parameter optimisations were done and selected based on development set performance. Over 500 hyper parameter setting  were evaluated and then the same setting was taken along with learning rate to OntoNotes dataset.

* for second Round - Independent hyper parameter searches on each dataset were carried out using Opportunity's implementation of particle swarm (better than random search). Over 500 hyper parameters were again searched this round - training failed occasionaly and gave large variations from run to run. hence top 5 setting were taken and 10 trials were done and selected the best which gave the best average performance.

* For CoNLL particle swarm produced best hyper parameters however for OntoNotes it didnot perform well.

* CoNLL model was trained for large number of epochs as the no sign of overfitting was observed. Contrary training on Onto Notes began to overfit after 18 epochs.

**Excluding Failed Trials** 

On CoNLL dataset BiLSTM model completed training without any difficulty, the BiLSTM-CNN model failed upto 10-15% of trials depending on the feature set. in OntoNotes it failed upto 1.5% trials. Using lower learning rate reduces the failure rate. Gradient Clipping and AdaDelta were effective in eliminating failures, however AdaDelta made training expensive. The threshold for CoNLL dataset was 95% and OntoNotes was 80% .

**Training and Tagging Speed** 

on Intel Xeon E5-2697 processor training takes about  6 hours while tagging takes about 12 seconds for CoNLL dataset. For OntoNotes dataset it was 10 hours and 60 seconds respectively.

**Results and Discussion** 

Given enough data the neural network automatically learns the relevant features for NER without feature engineering.

**Comparison with FFNN** by Collobert. Colloberts FFNN model was reinvented and compared with the BiLSTM model. FFNN was the baseline for comparison. FFNN was clearly inadequate for OntoNotes which proved that LSTM is required for a larger domain for NER.

**Character Level CNN's vs Character type and Capitalisation Features**

* BiLSTM -CNN models outperforms BiLSTM model significantly when given same feature set. 
* The effect is not statistically significant on OntoNotes when capitalisation features are added. 
* Character level CNN can replace handcrafted character features in some cases. When trained word embeddings were used large significant improvements were noted instead of random embedding regardless of additional features.
* 300 dimensional embedding present no significant improvement over 50 dimensional embedding
* gloVe embedding improved significantly over publicly available embeddings on CoNLL and word2vec skipgram embedding improved significantly over google's embedding on OntoNotes.

**Effect of DropOuts** 

Dropouts are essential for state of the art performance and improvement is statistically significant. DropOut is optimized on the devset, hence the chosen value may not be best performing.

**Lexicon Feature**

CoNLL using features from both SENNA lexicon and proposed DBpedia lexicon provides a significant improvement which is suspected to be because both the lexicons are complementary, SENNA lexicon is clean while DBpedia lexicon is noisy with higher coverage.

**Analysis on Onto Notes Performance** 

Model performs best on clean text like broadcast news (BN) and newswire(NW) and worst on noisy text like telephone conversation and webtext. The model also substantially improve except for telephone conversation. 

**Related Research**
* Recent approaches in NER has came from CRF, SVM and perceptron models performance is heavily dependent of feature engineering. k means clustering over a private database of search engine query logs  instead of phrase features helped to achieve 90.8% in CoNLL dataset. large scale unlabelled data was used to perform feature reduction this inturn acheived an F1 Score or 91.02%. 
* Training an NER system with entity linking has proved to be a success.
* **NER with Neural Networks** -  many approaches involve CRF model. However now we have computational power available hence complex neural networks are now being investigated for Neural Networks. CharWNN network augments neural network of Collobert with characterlevel  CNN's and reported improved performance on Spanish and Portugeese NER.
* BiLSTM has been used for POS tagging, chunking and NER task with heavy feature engineering instead of using CNN to extract character level features
* BiRNN with character level CNN's to perform German POS tagging is successfull
* Both word and character level BiLSTM has been used to create the current state of the art English POS tagging, Using BiLSTM instead of CNN allows for extraction of more sophisticated character level features but for NER it didnot perform significantly better than CNN and was computationaly expensive

**Conclusion**

Neural Network model with Bidirectional LSTM and character level CNN benefits from robust training  through dropout , achieved state of the art results in Named Entity Recognition with little feature engineering.



