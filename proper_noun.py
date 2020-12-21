# https://stackoverflow.com/questions/17669952/finding-proper-nouns-using-nltk-wordnet
import pandas as pd
import string

from collections import defaultdict
import random

import spacy
import re

import nltk

from nltk.tag import pos_tag
from nltk.corpus import wordnet

nlp = spacy.load('en_core_web_sm')


multiplier = 6
data_name = 'full_dataset_all' 

print(data_name)
data = pd.read_csv(data_name + '.csv', sep=';')

try:
    data = pd.read_csv(data_name + '.csv', sep=';')
except:
    data_name = input("Please type filename of data file: ")
    data = pd.read_csv(data_name + '.csv')

print(data.shape)
print(data.head())

data.fillna("")

tag_set = defaultdict(lambda:[])
syns = {}

tagged_txt = []

for i, row in data.iterrows():
    #print(row)
    split_txt = row['text'].split()
    
    
    if type(row['org_loc']) != list:
        split_1_txt = []

    if type(row['org_norp']) != list:
        split_2_txt = []
    

    tagged_sent = pos_tag(split_txt)
    tagged_sent1 = pos_tag(split_1_txt)
    tagged_sent2 = pos_tag(split_2_txt)

    propernouns = [word.translate(str.maketrans('', '', string.punctuation)) for word,pos in tagged_sent if pos == 'NNP']
    propernouns1 = [word.translate(str.maketrans('', '', string.punctuation)) for word,pos in tagged_sent1 if pos == 'NNP']
    propernouns2 = [word.translate(str.maketrans('', '', string.punctuation)) for word,pos in tagged_sent2 if pos == 'NNP']

    
    cross1 = set(propernouns1) & (set(propernouns))
    cross2 = set(propernouns2) & (set(propernouns))

    doc = nlp(row['text'])
    for sent in doc.sents:
        
        ents = [ent for ent in sent.ents]
        labels = [e.label_ for e in ents]

        for l,e in zip(labels,ents):
            tag_set[l] += [e]
    
    #for j in range(len(split_txt)):
    #    for syn in wordnet.synsets(split_txt[j]):
    #        r = syn.name()[0:syn.name().find(".")]
    #
    #        syns[syn] = r

       
for k,v in tag_set.items():
    s = set()

    vv = []
    for kk in v:
        tmp = kk.text.strip()
        tmp = tmp.strip('\n')
        tmp = tmp.strip('(')
        tmp = tmp.strip(')')

        #tmp = tmp.capitalize()
        
        vv += [tmp]

    tag_set[k] = set(vv)

# ['Michael','Jackson', 'McDonalds']

tag_set_pd = {}

max_len = 0

for k,v in tag_set.items():
    l = len(v)
    if l == 247:
        print("")
    if l > max_len:
        max_len = l

for k,v in tag_set.items():
    if len(v) != max_len:
        l = len(v)
        tag_set_pd[k] = list(v) + ["None"] * (max_len - len(v))
    else:
        tag_set_pd[k] = list(v)

ts_pd = pd.DataFrame.from_dict(tag_set_pd)


ts_pd.to_csv("tag_set.csv")
# manual removal?

def replace_entity(sent, ent, new_ent):
    
    if type(sent) != list:
        sent = sent.split()

    ent_spl = ent.text.split()
    new_ent = new_ent[0].split()

    
    for i, item in enumerate(sent):
        
        if item == ent_spl[0]:
            sent = sent[0:i] + new_ent + sent[i+len(ent):]
            break
        
        
    return sent     

#replace 
new_data = data.copy()


for _ in range(multiplier):
    for i, row in data.iterrows():

        new_text = row['text']

        new_sum1 = row['org_loc']
        new_sum2 = row['org_norp']
        
        doc = nlp(row['text'])

        if type(new_sum1) != str:
            new_sum1 = ""
        doc1 = nlp(new_sum1)

        if type(new_sum2) != str:
            new_sum2 = ""

        doc2 = nlp(new_sum2)

        for sent in doc.sents:
            
            ents = [ent for ent in sent.ents]
            labels = [e.label_ for e in ents]

            if type(new_text) != list:
                new_text = new_text.split()

            for e,l in zip(ents,labels):
                tag_s = tag_set[l]
                
                new_e =  random.sample(tag_s, 1)

                new_sum1 = ' '.join(replace_entity(new_sum1, e, new_e))
                new_sum2 = ' '.join(replace_entity(new_sum2, e, new_e))

                new_text = ' '.join(replace_entity(new_text, e, new_e))


        new_data.loc[-1] = [new_text, new_sum1, new_sum2]
        new_data.index = new_data.index + 1 

new_data.to_csv(data_name + '_' + str(multiplier) + 'X' + '_expanded.csv')

print("Done")