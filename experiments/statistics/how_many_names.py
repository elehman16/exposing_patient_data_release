import spacy
nlp = spacy.load('en_core_web_sm')

import pandas as pd
from tqdm import tqdm
names = pd.read_csv('setup_outputs/SUBJECT_ID_to_NAME.csv')
df = pd.read_csv('setup_outputs/SUBJECT_ID_to_NOTES_1a.csv')

fno = 0
lno = 0
both = 0
total = 0

for i in tqdm(range(len(df))):
    row = df.iloc[i]
    text = row.TEXT
    name_row = names[names['SUBJECT_ID'] == row.SUBJECT_ID]
    fn, ln = name_row['FIRST_NAME'].values[0].lower(), name_row['LAST_NAME'].values[0].lower()
    for txt in nlp(text).sents:
        txt = txt.text.lower()
        last_fn = fno
        last_ln = lno
        if len(fn) >= 2:
            fno += txt.count(fn)
    
        lno += txt.count(ln)
        if fno > last_fn and lno > last_ln:
            both += 1

        total += 1

print("All Name Mentions (including false positives):")
print('First Names: {}, Last Names: {}, Both: {}, Total: {}'.format(fno, lno, both, total))

fno = 0
lno = 0
total = 0
both = 0
either = 0

old_df = df
df = pd.read_csv('data/NOTEEVENTS.csv')
df = df[df.CATEGORY.isin(['Physician ','Nursing', 'Nursing/other', 'Discharge summary'])]
keys = set(old_df['SUBJECT_ID'].values)
split_df = df[df['SUBJECT_ID'].isin(keys)]
for idx, row in tqdm(split_df.iterrows(), total=len(split_df)):
    fc = row['TEXT'].count('[**Known firstname')
    lc = row['TEXT'].count('[**Known lastname')

    if fc > 0 and lc > 0:
        both += 1
    
    if fc > 0 or lc > 0:
        either += 1

    lno += lc
    fno += fc
    total += 1
    
print("Only True Positive Mentions:")
print('First Names: {}, Last Names: {}, Either: {}, Both: {}, Total: {}'.format(fno, lno, either, both, total))
