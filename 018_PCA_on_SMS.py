import pandas as pd
pd.options.display.width = 120

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

sms = pd.read_csv('nlpia_data/sms-spam.csv', index_col=False)
sms = sms[['spam', 'text']]

#adding an exclamation mark to the sms message index numbers to make them easier to spot
index = [f"sms{i}{'!'*j}" for (i,j) in zip(range(len(sms)), sms.spam)]
sms.index = index
#print(sms.head())

#calculate tf-idf vectors for each of these messages
tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()
#print(len(tfidf.vocabulary_))
tfidf_docs = pd.DataFrame(tfidf_docs)
#print(tfidf_docs.shape)
#print(sms.spam.sum())
"""
we have 4,837 SMS messages with 9,232 different 1-gram tokens
only 638 of these 4,837 messages (13%) are labeled
so we have an unbalanced training set with about 8:1 ham to spam
"""

"""
so we have many more unique words in our vocabulary (or lexicon) than 
we have SMS messages. And of those SMS messages only a small portion of 
them (1/8th) are labeled as spam. That’s a recipe for overfitting
"""

"""
Dimension reduction is the primary countermeasure for overfitting. By 
consolidating our dimensions (words) into a smaller number of 
dimensions (topics), our NLP pipeline will become more “general.” 
our spam filter will work on a wider range of SMS messages 
if you reduce our dimensions, or “vocabulary".
"""

#using PCA for SMS message semantic analysis
pca = PCA(n_components=16)
pca = pca.fit(tfidf_docs)
pca_topic_vectors = pca.transform(tfidf_docs)

columns = [f'topic{i}' for i in range(pca.n_components_)]
pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns=columns, index=index)
#print(pca_topic_vectors.round(3).head(6))

"""
If you’re curious about these topics, you can find out how much 
of each word they “contain” by examining their weights. By looking 
at the weights, you can see how often “half” occurs with the word 
“off” (as in “half off”) and then figure out which topic is your “discount” topic
"""
#examining weights
#print(pca.components_)

#assign words to all dimensions in pca transformation
#print(tfidf.vocabulary_) #contains all topics
column_nums, terms = zip(*sorted(zip(tfidf.vocabulary_.values(), tfidf.vocabulary_.keys())))

#pandas dataframe containing the weights, with labels for columns and rows
weights = pd.DataFrame(pca.components_, columns=terms, index=[f'topic{i}' for i in range(16)])
pd.options.display.max_columns = 8
#print(weights.head(4).round(3))

"""
some of those columns (terms) aren’t that interesting, so let’s explore our tfidf.vocabulary. 
Let’s see if we can find some of those “half off” terms and which topics they’re a part of
"""

pd.options.display.max_columns = 12
deals = weights['! ;) :) half off free crazy deal only $ 80 %'.split()].round(3)*100
#print(deals)
#print(deals.T.sum())

"""
Topics 4, 8, and 9 appear to all contain positive “deal” topic sentiment. And topics 0, 3, and 5 
appear to be “anti-deal” topics, messages about stuff that’s the opposite of “deals”: negative deals
"""