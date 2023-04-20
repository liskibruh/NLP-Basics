import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("better"))
print(lemmatizer.lemmatize("better", pos="a")) # a indicates adjective part of speech
print(lemmatizer.lemmatize("good", pos="a"))
print(lemmatizer.lemmatize("goods", pos="a"))
print(lemmatizer.lemmatize("goods", pos="n"))
print(lemmatizer.lemmatize("goodness", pos="n"))
print(lemmatizer.lemmatize("best", pos="a"))