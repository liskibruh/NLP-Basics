import numpy as np
import pandas as pd

sentence = """ Thomas Jefferson began building 
Monticello at the age of 26.
"""
#quick and dirty tokenizer
token_sequence = str.split(sentence)

#sorted so numbers come before letters,
#and capital letters come before lowercase letters
vocab = sorted(set(token_sequence))

num_tokens = len(token_sequence)
vocab_size = len(vocab)

#the empty table is as wide as your count of unique vocabulary terms
#and as high as the length of your document, 10 rows 10 columns
onehot_vectors = np.zeros((num_tokens, vocab_size), int)

#for each word in the sentence, 
#mark the column for that word in your vocabulary with a 1
for i, word in enumerate(token_sequence):
	onehot_vectors[i, vocab.index(word)] = 1

#we can just print onehot_vectors but for readability
#we'll convert the vector to pandas dataframe
df = pd.DataFrame(onehot_vectors, columns=vocab)

#replace all the zeros in the df by empty literal
df[df==0] = ''

print(df)