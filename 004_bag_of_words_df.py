import pandas as pd

sentences = "Thomas Construction Jefferson began building Monticello at the age of 26. \n"
sentences += "Construction was done mostly by local masons and carpenters. \n"
sentences += "He moved into the South Pavillion in 1770. \n"
sentences += "Turning Monticello into a neoclassical masterpiece was Jefferson's obsession."

corpus = {}

for i, sent in enumerate(sentences.split('\n')):
	corpus[f"sent{i}"] = dict((tok, 1) for tok in sent.split())

df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T
df = df.sort_index(axis=1)
print(df[df.columns[:10]])