import nltk
nltk.download('stopwords')

stop_words = nltk.corpus.stopwords.words('english')
print(f"Total stopwords: {len(stop_words)}")
print(stop_words[:7])

oneletter_stopwords = [sw for sw in stop_words if len(sw)==1]
print(oneletter_stopwords)