import spacy
import pandas as pd
from collections import defaultdict
from datasets import load_dataset


nlp = spacy.load('en_core_web_md')

datasets_dict = {
    'yelp_polarity': 'text'
}

# max_dataset_size = 2000
dataset_start_from = 800000
max_dataset_size = dataset_start_from + 500000
all_sents = list()

# Add Labels
for dataset_, text_field in datasets_dict.items():
    data = load_dataset(dataset_)
    print('total len', len(data['train']))
    for i, element in enumerate(data['train']):
        if i < dataset_start_from:
            continue
        if i % 1000 == 0:
            print(i, 'of', max_dataset_size)
        if i >= max_dataset_size:
            break
        text = element[text_field]
        doc = nlp(text)
        for sent in doc.sents:
            ents = [ent for ent in sent.ents]
            labels = list(set([e.label_ for e in ents]))
            if len(labels) > 2:
                labels = {e.label_: 1 for e in ents}
                text = {'text': sent.text}
                all_sents.append({**text, **labels})

# Show Co-occurrence matrix
df = pd.DataFrame(all_sents)
occ_df = df.iloc[:, df.columns != 'text']

occ_df = occ_df.fillna(0)
occ_df = occ_df.astype(int)
coocc = occ_df.T.dot(occ_df)
print(coocc.to_string())


# Generate dataset for annotation
# df2 = df.iloc[:, df.columns != 'text']
# df2 = df2.astype(int)
occ_df['text'] = df['text']
filtered_df = occ_df[(occ_df['ORG'] == 1) & (occ_df['NORP'] == 1) & (occ_df['LOC'] == 1)]
print('rows', filtered_df.size)
filtered_df.to_csv('./org-norp-loc-batch-4.csv')
print('hi')
