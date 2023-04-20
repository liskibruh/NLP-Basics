from collections import Counter
from nltk.tokenize import TreebankWordTokenizer
import nltk
nltk.download('stopwords', quiet=True)

tokenzier = TreebankWordTokenizer()

#load text from txt file
with open("nlpia_data/kite_intro.txt") as f:
    kite_text = f.read()
    
tokens = tokenzier.tokenize(kite_text.lower())
token_counts = Counter(tokens)
#print(token_counts)

stopwords = nltk.corpus.stopwords.words('english')
tokens = [x for x in tokens if x not in stopwords]

kite_counts = Counter(tokens)

#vectorizing
document_vector = []
doc_length = len(tokens)

for _, value in kite_counts.most_common():
    document_vector.append(value/doc_length)
    
print(document_vector)

