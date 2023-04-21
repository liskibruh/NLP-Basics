from collections import OrderedDict, Counter
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

#read text from file
with open("nlpia_data/kite_intro.txt","r") as f:
    kite_intro = f.read()
    
with open("nlpia_data/kite_history.txt","r") as f:
    kite_history = f.read()
    

kite_intro = kite_intro.lower()
kite_history = kite_history.lower()

#tokenize both documents
intro_tokens = tokenizer.tokenize(kite_intro)
history_tokens = tokenizer.tokenize(kite_history)

#total tokens count of both docs
intro_total = len(intro_tokens)
history_total = len(history_tokens)

#term frequency of 'kite' word in both docs
intro_tf = {}
history_tf = {}

intro_counts = Counter(intro_tokens)
intro_tf['kite'] = intro_counts['kite']/intro_total

history_counts = Counter(history_tokens)
history_tf['kite'] = history_counts['kite']/history_total

print(f"Term frequency of 'kite' in intro is: {intro_tf['kite']:.4f}")
print(f"Term frequency of 'kite' in history is: {history_tf['kite']:.4f}")

#how these numbers relate to 'and' word
intro_tf['and'] = intro_counts['and']/intro_total
history_tf['and'] = history_counts['and']/history_total

print(f"Term frequency of 'and' in intro is: {intro_tf['and']:.4f}")
print(f"Term frequency of 'and' in history is: {history_tf['and']:.4f}")

#term frequency of 'china' word in both docs
intro_tf['china'] = intro_counts['china']/intro_total
history_tf['china'] = history_counts['china']/history_total

print(f"Term frequency of 'china' in intro is: {intro_tf['china']:.4f}")
print(f"Term frequency of 'china' in history is: {history_tf['china']:.4f}")

num_docs_containing_kite = 0
num_docs_containing_and = 0
num_docs_containing_china = 0
for doc in [intro_tokens, history_tokens]:
    if 'kite' in doc:
        num_docs_containing_kite += 1
for doc in [intro_tokens, history_tokens]:
    if 'and' in doc:
        num_docs_containing_and += 1
for doc in [intro_tokens, history_tokens]:
    if 'china' in doc:
        num_docs_containing_china += 1

        
print(num_docs_containing_kite, num_docs_containing_and, num_docs_containing_china)

#IDF for all three words
num_docs = 2
intro_idf = {}
history_idf = {}

intro_idf['and'] = num_docs/num_docs_containing_and
history_idf['and'] = num_docs/num_docs_containing_and

intro_idf['kite'] = num_docs/num_docs_containing_kite
history_idf['kite'] = num_docs/num_docs_containing_kite

intro_idf['china'] = num_docs/num_docs_containing_china
history_idf['china'] = num_docs/num_docs_containing_china

#tfidf for intro doc
intro_tfidf = {}
intro_tfidf['and'] = intro_tf['and'] * intro_idf['and']
intro_tfidf['kite'] = intro_tf['kite'] * intro_idf['kite']
intro_tfidf['china'] = intro_tf['china'] * intro_idf['china']

#tfidf for history doc
history_tfidf = {}
history_tfidf['and'] = history_tf['and'] * history_idf['and']
history_tfidf['kite'] = history_tf['kite'] * history_idf['kite']
history_tfidf['china'] = history_tf['china'] * history_idf['china']

print(intro_tfidf)
print(history_tfidf)