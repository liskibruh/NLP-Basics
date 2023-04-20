from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sa = SentimentIntensityAnalyzer()

all_tokens_and_scores = sa.lexicon
#print(all_tokens_and_scores)
spaces_and_ngrams = [(tok, score) for tok, score in sa.lexicon.items() if " " in tok ]
#print(spaces_and_ngrams)

pol_scores = sa.polarity_scores(text = "Python is not a bad choice for most applications.")
#print(pol_scores)

corpus = ["Absolutely perfect! Love it! :-) :-)",
          "Horrible! Completely useless. :(",
          "It was OK. Some good and some bad things."]

for doc in corpus:
    scores  = sa.polarity_scores(doc)
    print(f"{scores['compound']:+}: {doc}")