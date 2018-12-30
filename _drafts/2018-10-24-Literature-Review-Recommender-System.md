**Why we need recommender system ?**

​	Web applications would love to give curated content to the user so that he has a good quality of options visible to him/her while using the web application. 

*eg:-* 

Scenario 1 -*Customized News articles - a news agency website would want to give customized news to each of its user based on the reading habits of each user*. 

Scenario 2 - *Based on past history data what a customer would like to buy*

**Types of Recommender Systems**

Content based systems - *examine properties of the items recommended*

Collaborative filtering systems - *recommend items based on similarity measures between users and/or items.*

**Utility Matrix**

​	In a recommender system, we are usualy trying to identify potential relationship between two classes namely  *User* and *item*. The Data showing the relationship between these two classes (*User -row and Item in col*)  is called a Utility Matrix. Utility matrix are generally *Sparse Matrix* (Matrix which has a value only when a relationship exist between both entities, hence most elements are zero in a sparse matrix

​	Eventhough the goal of a recommender system is to predict the blanks in the utility matrix,  it is not necessary to predict every blank, it is necessary only to predict certain items that have a very high probablity.

**Long Tail** *(Marketing Theory)*

​	Physical delivery systems are characterized by a scarcity of resources. Brick-and-mortar stores have limited shelf space, and can show the customer only a small fraction of all the choices that exist. Hence in real world stores it is not possible to tailor each store based on the choice of each individual customer. However this is not the case with an online store, they have virtually infinite store space. And hence an online store showcase rare products that are not necessarily provided in custom physical store (because of less demand / more shelf space). But this long tail phenomenon leaves an online user with another problem the screen space in a laptop / mobile with which a user surfs the website is limited hence the retailer should be meticulous that he is providing items that intrests the user and guess who helps a retailer to solve this problem, yep Recommender systems to your help. !

**Content Based Recommendations**

This approach as mentioned previously focus on the properties of the item hence an *Item Profile* becomes necessary. *eg:- Movie Genre is an important item in the Item profile of a movie*. If the data that we are working is a document or an image the process becomes tricky. like movie genre for movies we need to have a topic for a document, so that we can recommend document based on topics. 

**Topic detection using TFIDF**  will become handy here.  we first remove the stop words from the document and then compute the TFIDF score of each word in the document and the onces with highest frequency characterizes the document. 

Now if the data is images how do we get the profile of an image ? **Tagging Of Images** comes into picture here. This is still a research area perhaps i would comeback later to explore on the methods of tagging images.

So the important aspect of a content based recommendation is the generation of profile vectors for both user and item there be we can match items to users as well as users to item both ways depending the need. Once the profile vectors are created, we could easily find out the similarity based on  distance metrics like cosine distance.

**Collaborative Filtering**

In collaborative filtering  we focus on the similarity of one user vector (each single row in the sparse matrix). if the user vectors are close according to some distance criteria we assume that users are similar. Recommendation for a user is made by looking at the user that are more similar to U in this sense. The process of identifying similar user and then recommending what similar users like is called collaborative filtering.



Reference :- 

[Recommender systems]: http://infolab.stanford.edu/~ullman/mmds/ch9.pdf	"Recommender systems"
























