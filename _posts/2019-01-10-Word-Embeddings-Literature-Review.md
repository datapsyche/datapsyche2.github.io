## Word Embeddings - Literature Review

- Word embeddings are representations of a document library capable of capturing context of a word in a document, semantic and syntactic similarity, relation with other words.
- Vector representation of a particular word.

**How to create a word Embedding ?** - **Word2vec** is the most popular technique to learn word embeddings, it is using shallow neural network.

**Why do we need Word Embedding ?** - In One hot encoding all the words are independent of each other. Word Embedding aims to quantify and categorize semantic similarities between linguistic items based on their distributional properties in large samples of data. The underlying principle is that a word is characterized by the company it keeps.

### Paper Reading - Evaluation of  sentence embedding in downstream and linguistic probing tasks.

*by Christian S Perone, Roberto Silvera, Thomas Paula*

Word Embedding ranges from Neural Probablistic Language Model, Word2vec, GloVe, and ELMO. Most of them rely on distributional linguistic hypothesis but differ on assumption of how meaning or context are modeled to produce the word embeddings. Word embeddings provide high quality representation for words but representing sentences, paragraphs is still an open research.

Most common approach is **Bag of Words** model. Here we create a Bag of words word vector using simple arithmetic mean of the embeddings for the words in a sentence along the words dimension. Though the approach seems trivial and limited lot of improvements have taken place in Bag of Words model by using weighted averages and modifying them using singular value decomposition (SVD) the method is called **smooth inverse frequency**.

word embedding based on Encoder / Decoder architectures - **Skip Thought** - a skip gram model from word2vec was abstracted to form sentence level encoder trained on self supervised fashion. 







