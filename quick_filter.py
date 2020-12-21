import pandas as pd
import textdistance

existing_csv = '/Users/bradwindsor/classwork/natural_language_processing/paper/existing.csv'
existing = pd.read_csv(existing_csv)

draft_csv = '/Users/bradwindsor/classwork/natural_language_processing/paper/draft_candidates.csv'
draft = pd.read_csv(draft_csv)

output = []
print('draft length', draft.size)
for j, row in draft.iterrows():
    max_sim = float('-inf')
    print('i', j)
    for i, existing_row in existing.iterrows():
        sim = textdistance.hamming.normalized_similarity(row['text'], existing_row['text'])
        max_sim = max(sim, max_sim)

    if max_sim < 0.6:
        output.append(row['text'])

print('new length', len(output))

