import nltk
nltk.download('stopwords')
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words

stop_words = nltk.corpus.stopwords.words('english')
print(f"Number of NLTK stopwords = {len(stop_words)}")
print(f"Number of sklearn stopwords = {len(sklearn_stop_words)}")

stop_words=set(stop_words)

#total unique stopwords in both packages
total_unique_both = len(stop_words.union(sklearn_stop_words))
print(f"Total unique stopwords in both packages: {total_unique_both}")

#common stopwords in both packages
common_stopwords = len(stop_words.intersection(sklearn_stop_words))
print(f"Common stopwords in both packages: {common_stopwords}")
