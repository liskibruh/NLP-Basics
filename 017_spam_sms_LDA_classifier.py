import pandas as pd
pd.options.display.width = 120

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
from sklearn.preprocessing import MinMaxScaler

sms = pd.read_csv('nlpia_data/sms-spam.csv', index_col=False)
sms = sms[['spam', 'text']]

tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf_model.fit_transform(raw_documents=sms.text).toarray()

#print(tfidf_docs[0:5])
#print(tfidf_docs.shape) #(4837, 9232) 9,232 words in vocabulary, almost twice as many words as messages
#print(sms.spam.sum()) #638 are spam

"""
The nltk.casual_tokenizer gave you 9,232 words in your vocabulary. 
You have almost twice as many words as you have messages. 
And you have almost ten times as many words as spam messages. 
So your model won’t have a lot of information about the words that will 
indicate whether a message is spam or not. Usually, a Naive Bayes
classifier won’t work well when your vocabulary is much larger than 
the number of labeled examples in your dataset. 
That’s where the semantic analysis techniques can help.

Let’s start with the simplest semantic analysis technique, LDA. 
You could use the LDA model in sklearn.discriminant_analysis.
LinearDiscriminantAnalysis. But you only need compute the centroids of your 
binary class (spam and nonspam) in order to “train” this model, 
so you’ll do that directly:
"""

#can use this mask to select only spam rows from a numpy array or pandas dataframe
mask = sms.spam.astype(bool).values

#because the tfidf vectors are rwo vectors, we need to make sure
#numpy computes the mean for each columnindependently using axis=0
spam_centroid = tfidf_docs[mask].mean(axis=0)
ham_centroid = tfidf_docs[~mask].mean(axis=0)

#print(spam_centroid.round(2))
#print(spam_centroid.round(2))

#now we can subtract one centroid from the other to get the line betwen them
spamminess_score = tfidf_docs.dot(spam_centroid - ham_centroid)

#print(spamminess_score.round(2))

"""
This raw spamminess_score is the distance along the line from the 
ham centroid to the spam centroid. We calculated that score by projecting 
each TF-IDF vector onto that line between the centroids using the dot product.

Ideally, we’d like our score to range between 0 and 1, like a probability. 
The sklearnMinMaxScaler can do that for us
"""
sms['lda_score'] = MinMaxScaler().fit_transform(\
    spamminess_score.reshape(-1,1))

sms['lda_predict'] = (sms.lda_score > .5).astype(int)
#print(sms['spam lda_predict lda_score'.split()].round(2).head(6))

#accuracy score = 0.977
print((1. - (sms.spam - sms.lda_predict).abs().sum() / len(sms)).round(3))
