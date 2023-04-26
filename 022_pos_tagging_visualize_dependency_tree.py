import spacy
from spacy.displacy import render
en_model = spacy.load('en_core_web_md')

sentence = 'In 1541 Desoto wrote in his journal about the Pascagoula.'
parsed_sent = en_model(sentence)

with open('pascagoula.html', 'w') as f:
    f.write(render(docs=parsed_sent, page=True, options=dict(compact=True)))