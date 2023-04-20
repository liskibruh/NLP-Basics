stop_words = ['a', 'an', 'the', 'on', 'of', 'off', 'this', 'is']
tokens = ['the', 'house', 'is', 'on', 'fire']

tokens_without_stopwords = [x for x in tokens if x not in stop_words]
print(tokens_without_stopwords)