import pandas as pd
import spacy

nlp = spacy.load('en_core_web_md')

data = pd.read_csv('/Users/bradwindsor/classwork/natural_language_processing/paper/full_dataset_all.csv', engine='python')
# data = data[['text', 'ORG NORP Simplified']]
data = data.fillna('')

other_norp =['Mexican', 'French', 'Vienamese', 'Italian', 'Korean', 'Chinese', 'Carribean', 'Thai', 'Asian', 'Mediterranean']
other_norp = other_norp + [o.lower() for o in other_norp]

output = []
for i, row in data.iterrows():
    source_text = row['text']
    org_norp_simp = row['org_norp']
    org_loc_simp = row['org_loc']
    doc = nlp(source_text)
    orgs = [e for e in doc.ents if e.label_ == 'ORG']
    norps = [e for e in doc.ents if e.label_ == 'NORP']
    locs = [e for e in doc.ents if e.label_ in ('LOC', 'GPE')]
    for j, ent in enumerate(orgs):
        for ent_text in [ent.text, ent.text.lower(), ent.text.upper(), ent.text[0].upper() + ent.text[1:].lower()]:
            source_text = source_text.replace(ent_text, f'ORG{j}')
            org_norp_simp = org_norp_simp.replace(ent_text, f'ORG{j}')
            org_loc_simp = org_loc_simp.replace(ent_text, f'ORG{j}')
    for j, ent in enumerate(norps):
        for ent_text in [ent.text, ent.text.lower(), ent.text.upper(), ent.text[0].upper() + ent.text[1:].lower()]:
            source_text = source_text.replace(ent_text, f'NORP{j}')
            org_norp_simp = org_norp_simp.replace(ent_text, f'NORP{j}')
            org_loc_simp = org_loc_simp.replace(ent_text, f'NORP{j}')
    for j, ent in enumerate(locs):
        for ent_text in [ent.text, ent.text.lower(), ent.text.upper(), ent.text[0].upper() + ent.text[1:].lower()]:
            source_text = source_text.replace(ent_text, f'LOC{j}')
            org_norp_simp = org_norp_simp.replace(ent_text, f'LOC{j}')
            org_loc_simp = org_loc_simp.replace(ent_text, f'LOC{j}')


    max_norp_found = len(norps)
    for norp in other_norp:
        if norp in source_text or norp in org_norp_simp:
            max_norp_found += 1
            source_text = source_text.replace(norp, f'NORP{max_norp_found}')
            org_norp_simp = org_norp_simp.replace(norp, f'NORP{max_norp_found}')
            org_loc_simp = org_loc_simp.replace(norp, f'NORP{max_norp_found}')


    output.append([source_text, org_norp_simp, org_loc_simp])

created = pd.DataFrame(output, columns=['text', 'org_norp', 'org_loc'])
created.to_csv('/Users/bradwindsor/classwork/natural_language_processing/paper/masked_dataset_all.csv')