from nltk.util import ngrams
import re

sentence = "Thomas Jefferson began builidng Monticello at the age of 26."
pattern = re.compile(r"([-\s.,;!?])+")
tokens = pattern.split(sentence)
tokens = [x for x in tokens if x and x not in '- \t\n.,;!?']

#bigram/2-grams tokenizer
two_grams = list(ngrams(tokens,2))
#coverting the gram tuples to a flat list
two_grams = [" ".join(x) for x in two_grams]

print(two_grams)