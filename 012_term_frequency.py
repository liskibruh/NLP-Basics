from nltk.tokenize import TreebankWordTokenizer
from collections import Counter

sentence = """The faster Harry got to the store, the faster Harry, 
    the faster, would get home."""

tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(sentence.lower())
#print(tokens)

bag_of_words = Counter(tokens)
#print(bag_of_words)
#print(bag_of_words.most_common(4))

#calculate normalized term frequency for a certain word
times_harry_appears = bag_of_words['harry']
num_unique_words = len(bag_of_words)
tf = times_harry_appears/num_unique_words
print(f"Term Frequency for 'Harry': {tf}")