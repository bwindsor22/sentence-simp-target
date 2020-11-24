import spacy
import pandas as pd
from collections import defaultdict
from datasets import load_dataset


nlp = spacy.load('en_core_web_md')

datasets_dict = {
    'yelp_polarity': 'text'
}

all_sents = list()


for dataset_, text_field in datasets_dict.items():
    data = load_dataset(dataset_)
    for i, element in enumerate(data['train']):
        text = element[text_field]
        doc = nlp(text)
        for sent in doc.sents:
            ents = [ent for ent in sent.ents]
            labels = list(set([e.label_ for e in ents]))
            if len(labels) > 2:
                labels = {e.label_: 1 for e in ents}
                text = {'text': sent.text}
                all_sents.append({**text, **labels})

df = pd.DataFrame(all_sents)
df2 = df.iloc[:, df.columns != 'text']
corr = df2.corr()

df_asint = df.astype(int)
coocc = df_asint.T.dot(df_asint)


print('hi')


