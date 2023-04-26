import spacy
en_model = spacy.load('en_core_web_md')

sentence = ("In 1541 Desoto wrote in his journal that the Pascagoula people" + \
    "ranged as far north as the confluence of the Leaf and Chickasawhay rivers at 30.4, -88.5.")

parsed_sent = en_model(sentence)
#print(parsed_sent.ents)

print(' '.join([f'{tok}_{tok.tag_}' for tok in parsed_sent]))

"""
'In_IN': 'In' is a preposition (IN).
'1541_CD': '1541' is a cardinal number (CD).
'Desoto_NNP': 'Desoto' is a proper noun (NNP).
'wrote_VBD': 'wrote' is a verb in the past tense (VBD).
'in_IN': 'in' is a preposition (IN).
'his_PRP$': 'his' is a possessive pronoun (PRP$).
'journal_NN': 'journal' is a singular noun (NN).
'that_IN': 'that' is a conjunction (IN).
'the_DT': 'the' is a determiner (DT).
'Pascagoula_NNP': 'Pascagoula' is a proper noun (NNP).
'people_NNS': 'people' is a plural noun (NNS).
"""

