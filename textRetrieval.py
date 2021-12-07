#!/usr/bin/env python
# coding: utf-8

# ### Text retrieval
# 
# This guide will introduce techniques for organizing text data. It will show how to analyze a large corpus of text, extracting feature vectors for individual documents, in order to be able to retrieve documents with similar content.
# 
# [scipy](https://www.scipy.org/) and [scikit-learn](scikit-learn.org) are required to run through this document, as well as a corpus of text documents. This code can be adapted to work with a set of documents you collect. For the purpose of this example, we will use the well-known Reuters-21578 dataset with 90 categories. To download this dataset, download it manually from [here](http://disi.unitn.it/moschitti/corpora.htm), or run `download.sh` in the `data` folder (to get all the other data for ml4a-guides as well), or just run:

import os
# Once you've downloaded and unzipped the dataset, take a look inside the folder. It is split into two folders, "training" and "test". Each of those contains 91 subfolders, corresponding to pre-labeled categories, which will be useful for us later when we want to try classifying the category of an unknown message. In this notebook, we are not worried about training a classifier, so we'll end up using both sets together.
# 
# Let's note the location of the folder into a variable `data_dir`.

data_dir = './Reuters21578-Apte-115Cat'


# Let's open up a single message and look at the contents. This is the very first message in the training folder, inside of the "acq" folder, which is a category apparently containing news of corporate acquisitions.

post_path = os.path.join(data_dir, "training", "acq", "0000005")
with open (post_path, "r") as p:
    raw_text = p.read()
    print(raw_text)


# Our collection contains over 15,000 articles with a lot of information. It would take way too long to get through all the information.

# this gives us all the groups (from training subfolder, but same for test)
groups = [g for g in os.listdir(os.path.join(data_dir, "training")) if os.path.isdir(os.path.join(data_dir, "training", g))]
print(groups)


# Let's load all of our documents (news articles) into a single list called `docs`. We'll iterate through each group, grab all of the posts in each group (from both training and test directories), and add the text of the post into the `docs` list. We will make sure to exclude duplicate posts by cheking if we've seen the post index before.

import re

docs = []
post_idx = []
for g, group in enumerate(groups):
    if g%10==0:
        print ("reading group %d / %d"%(g+1, len(groups)))
    posts_training = [os.path.join(data_dir, "training", group, p) for p in os.listdir(os.path.join(data_dir, "training", group)) if os.path.isfile(os.path.join(data_dir, "training", group, p))]
    posts_test = [os.path.join(data_dir, "test", group, p) for p in os.listdir(os.path.join(data_dir, "test", group)) if os.path.isfile(os.path.join(data_dir, "test", group, p))]
    posts = posts_training + posts_test
    for post in posts:
        idx = post.split("/")[-1]
        if idx not in post_idx:
            post_idx.append(idx)
            with open(post, "r") as p:
                raw_text = p.read()
                raw_text = re.sub(r'[^\x00-\x7f]',r'', raw_text) 
                docs.append(raw_text)

print("\nwe have %d documents in %d groups"%(len(docs), len(groups)))
#print("\nhere is document 100:\n%s"%docs[100])



# We will now use `sklearn`'s `TfidfVectorizer` to compute the tf-idf matrix of our collection of documents. The tf-idf matrix is an `n`x`m` matrix with the `n` rows corresponding to our `n` documents and the `m` columns corresponding to our terms. The values corresponds to the "importance" of each term to each document, where importance is *. In this case, terms are just all the unique words in the corpus, minus english _stopwords_, which are the most common words in the english language, e.g. "it", "they", "and", "a", etc. In some cases, terms can be n-grams (n-length sequences of words) or more complex, but usually just words.
# 
# To compute our tf-idf matrix, run:
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
tfidf = vectorizer.fit_transform(docs)

print(tfidf)


# We see that the variable `tfidf` is a sparse matrix with a row for each document, and a column for each unique term in the corpus. 
# 
# Thus, we can interpret each row of this matrix as a feature vector which describes a document. Two documents which have identical rows have the same collection of words in them, although not necessarily in the same order; word order is not preserved in the tf-idf matrix. Regardless, it seems reasonable to expect that if two documents have similar or close tf-idf vectors, they probably have similar content.
doc_idx = 5

doc_tfidf = tfidf.getrow(doc_idx)
all_terms = vectorizer.get_feature_names()
terms = [all_terms[i] for i in doc_tfidf.indices]
values = doc_tfidf.data

print(docs[doc_idx])
print("document's term-frequency pairs:")
print(", ".join("\"%s\"=%0.2f"%(t,v) for t,v in zip(terms,values)))


# In practice however, the term-document matrix alone has several disadvantages. For one, it is very high-dimensional and sparse (mostly zeroes), thus it is computationally costly. 
# 
# Additionally, it ignores similarity among groups of terms. For example, the words "seat" and "chair" are related, but in a raw term-document matrix they are separate columns. So two sentences with one of each word will not be computed as similarly.
# 
# One solution is to use latent semantic analysis (LSA, or sometimes called latent semantic indexing). LSA is a dimensionality reduction technique closely related to principal component analysis, which is commonly used to reduce a high-dimensional set of terms into a lower-dimensional set of "concepts" or components which are linear combinations of the terms.
# 
# To do so, we use `sklearn`'s `TruncatedSVD` function which gives us the LSA by computing a [singular value decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition) of the tf-idf matrix. 
from sklearn.decomposition import TruncatedSVD

lsa = TruncatedSVD(n_components=100)
tfidf_lsa = lsa.fit_transform(tfidf)


# How to interpret this? `lsa` holds our latent semantic analysis, expressing our 100 concepts. It has a vector for each concept, which holds the weight of each term to that concept. `tfidf_lsa` is our transformed document matrix where each document is a weighted sum of the concepts. 
# 
# In a simpler analysis with, for example, two topics (sports and tacos), one concept might assign high weights for sports-related terms (ball, score, tournament) and the other one might have high weights for taco-related concepts (cheese, tomato, lettuce). In a more complex one like this one, the concepts may not be as interpretable. Nevertheless, we can investigate the weights for each concept, and look at the top-weighted ones. For example, here are the top terms in concept 1.
components = lsa.components_[1]
all_terms = vectorizer.get_feature_names()

idx_top_terms = sorted(range(len(components)), key=lambda k: components[k])

print("10 highest-weighted terms in concept 1:")
for t in idx_top_terms[:10]:
    print(" - %s : %0.02f"%(all_terms[t], t))


# The top terms in concept 1 appear related to accounting balance sheets; terms like "net", "loss", "profit".
# 
# Now, back to our documents. Recall that `tfidf_lsa` is a transformation of our original tf-idf matrix from the term-space into a concept-space. The concept space is much more valuable, and we can use it to query most similar documents. We expect that two documents which about similar things should have similar vectors in `tfidf_lsa`. We can use a simple distance metric to measure the similarity, euclidean distance or cosine similarity being the two most common. 
# 
# Here, we'll select a single query document (index 300), calculate the distance of every other document to our query document, and take the one with the smallest distance to the query.
from scipy.spatial import distance

query_idx = 400

# take the concept representation of our query document
query_features = tfidf_lsa[query_idx]

# calculate the distance between query and every other document
distances = [ distance.euclidean(query_features, feat) for feat in tfidf_lsa ]
    
# sort indices by distances, excluding the first one which is distance from query to itself (0)
idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:]

# print our results
query_doc = docs[query_idx]
return_doc = docs[idx_closest[0]]
print("QUERY DOCUMENT:\n %s \nMOST SIMILAR DOCUMENT TO QUERY:\n %s" %(query_doc, return_doc))



# Interesting find! Our query document appears to be about tax incentives for domestic oil and natural gas. Our return document is a related article about the same topic. Try looking at the next few closest results. A quick inspection reveals that most of them are about the same story.
# 
# Thus we see the value of this procedure. It gives us a way to quickly identify articles which are related to each other. This can greatly aide journalists who have to sift through a lot of content which is not always indexed or organized usefully. 
# 
# More creatively, we can think of other ways this can be made useful. For example, what if instead of making our documents the articles themselves, what if they were made to be paragraphs from the articles? Then, we could potentially discover relevant paragraphs about one topic which are buried in an article which is otherwise about a different topic. We can combine this with handcrafted filters as well (date range, presence of a word or name, etc); perhaps you want to quickly find every quote politican X has made about topic Y. This provides an effective means to do so.


#Generating Wordcloud for first 100 docs.
text_data=[]
for i in range(500):
	text=docs[i]
	text_data.append(text)


from wordcloud import WordCloud, STOPWORDS
from nltk import flatten
import matplotlib.pyplot as plt

#text = ' '.join(flatten(text_data)) ## text string of cleaned tokens 
text = ' '.join(text_data)
stopwords = set(STOPWORDS)
stopwords.update(["said","will","dlr","pct","Co","pesch"])
wordcloud = WordCloud( stopwords=stopwords, background_color="black",width=2800,height=2000).generate(text)

##Plotting
#plt.figure(figsize=(20,10))
plt.axis("off")

# "bilinear" interpolation is used to make the displayed image appear more smoothly without overlapping of words  
# can change to some other interpolation technique to see changes like "hamming"
plt.imshow(wordcloud, interpolation="bilinear") 

plt.show()


