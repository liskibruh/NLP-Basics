import copy
from collections import OrderedDict, Counter
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

#corpus containing three docs
docs = ["the faster Harry got to the store,\
        the faster and faster Harry would get home."]
docs.append("Harry is hairy and faster than Jill.")
docs.append("Jill is not as hairy as Harry.")

doc_tokens = []
for doc in docs:
    doc_tokens += [sorted(tokenizer.tokenize(doc.lower()))]

#print([len(each_doc_tokens) for each_doc_tokens in doc_tokens])

all_doc_tokens = sum(doc_tokens, [])
#print(len(all_doc_tokens))

lexicon = sorted(set(all_doc_tokens))

"""
Each of your three document vectors will need to have 18 values, 
even if the document for that vector doesn’t contain all 18 
words in your lexicon
"""
zero_vector = OrderedDict((token,0) for token in lexicon)
#print(zero_vector)

"""
Now you’ll make copies of that base vector, update the values of 
the vector for each document, and store them in an array
"""
doc_vectors = []
for doc in docs:
    vec = copy.copy(zero_vector)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)
    for key, value in token_counts.items():
        vec[key] = value/len(lexicon)
    doc_vectors.append(vec)