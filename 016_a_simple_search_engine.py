import copy
import math
from collections import OrderedDict, Counter
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

def cosine_sim(vec1, vec2):
    """Lets convert our dictionaries to lists for easier matching."""
    vec1 = [val for val in vec1.values()]
    vec2 = [val for val in vec2.values()]
    
    dot_prod = 0
    for i, v in enumerate(vec1):
        dot_prod += v*vec2[i]
    
    mag_1 = math.sqrt(sum([x**2 for x in vec1]))
    mag_2 = math.sqrt(sum([x**2 for x in vec2]))
    
    return dot_prod/(mag_1 * mag_2)


#corpus containing three docs
docs = ["the faster Harry got to the store,\
        the faster and faster Harry would get home."]
docs.append("Harry is hairy and faster than Jill.")
docs.append("Jill is not as hairy as Harry.")

doc_tokens = []
for doc in docs:
    doc_tokens += [sorted(tokenizer.tokenize(doc.lower()))]

all_doc_tokens = sum(doc_tokens, [])
lexicon = sorted(set(all_doc_tokens))

"""
Each of your three document vectors will need to have 18 values, 
even if the document for that vector doesn’t contain all 18 
words in your lexicon
"""
zero_vector = OrderedDict((token,0) for token in lexicon)

document_tfidf_vectors = []
for doc in docs:
    vec = copy.copy(zero_vector)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)
    
    for key, value in token_counts.items():
        docs_containing_key = 0
        for _doc in docs:
            if key in _doc:
                docs_containing_key += 1
        tf = value / len(lexicon)
        if docs_containing_key:
            idf = len(docs)/docs_containing_key
        else:
            idf = 0
        vec[key] = tf*idf
    document_tfidf_vectors.append(vec)



query = "How long does it take to get to the store?"
query_vec = copy.copy(zero_vector)
query_vec = copy.copy(zero_vector)

tokens = tokenizer.tokenize(query.lower())
tokens_counts = Counter(tokens)

for key, value in tokens_counts.items():
    docs_containing_key = 0
    for _doc in docs:
        if key in _doc.lower():
            docs_containing_key += 1
    if docs_containing_key == 0:
        continue
    
    tf = value/len(tokens)
    idf = len(docs)/docs_containing_key
    query_vec[key] = tf*idf
    
result1=cosine_sim(query_vec, document_tfidf_vectors[0])
result2=cosine_sim(query_vec, document_tfidf_vectors[1])
result3=cosine_sim(query_vec, document_tfidf_vectors[2])

print(result1)
print(result2)
print(result3)

"""
The prints are:
0.6132857433407973
0.0
0.0

you can safely say document 0 has the most relevance for your query!
"""