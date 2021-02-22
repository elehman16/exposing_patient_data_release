import glob
import config
import spacy 
import pandas as pd 
import re
from tqdm import tqdm

name_of_interest = ''
loc = '/home/eric/dis_rep/physionet/files/mimiciii/physionet.org/files/deid/1.1/lists/last_names_popular.txt'
#tfl = '/home/eric/exposing_patient_data_v2/extracted_data/*.txt'
tfl = '/home/eric/exposing_patient_data_v2/bert-base-uncased/*.txt'
nlp = spacy.load('en')
nlp.max_length = 10e7

def get_all_relevant_conditions(name):
    pass

def find_all(s, str_):
    return [m.start() for m in re.finditer(s, str_)]

def ner_tag_names(txt):
    doc = nlp(txt)
    counts = {}
    for tok in doc:
        if tok.pos_ == 'PROPN':
            counts[tok.text] = counts[tok.text] + 1 if tok.text in counts else 1
    
    return counts

def create_counts(txt):
    l = txt.split(' ')
    dic_ = {}
    for w in l:
        if w in dic_:
            dic_[w] += 1
        else: 
            dic_[w] = 1

    return dic_ 

# Read in the dataframes
df = pd.read_csv(config.SUBJECT_ID_to_NAME)
modified = set(pd.read_csv(config.MODIFIED_SUBJECT_IDS)['SUBJECT_ID'])

# Lower case the first and last names
df['FIRST_NAME'] = df['FIRST_NAME'].apply(lambda x: str(x).lower())
df['LAST_NAME'] = df['LAST_NAME'].apply(lambda x: str(x).lower())

# Get a list of names the model has/hasn't seen.
names = df[df['SUBJECT_ID'].isin(modified)] 
non_names = df[~df['SUBJECT_ID'].isin(modified)]

# Get initial counts
non_names_counter = dict.fromkeys(filter(lambda x: len(x) > 0, non_names['LAST_NAME'].values), 0)
lname_counter = dict.fromkeys(filter(lambda x: len(x) > 0, names['LAST_NAME'].values), 0)

txt_files = glob.glob(tfl)
for f in tqdm(txt_files):
    with open(f) as tmp: txt = tmp.read()
    words = ner_tag_names(txt)
    #words = create_counts(txt)
    for n in lname_counter:
        if n in words: 
            lname_counter[n] += words[n]
   
    for n in non_names_counter:
        if n in words:
            non_names_counter[n] += words[n]

cin = 0
for n in lname_counter.keys():
    if lname_counter[n] > 0:
        cin += 1

cout = 0
for n in non_names_counter:
    if non_names_counter[n] > 0:
        cout += 1

print(f"Percent of names mentioned that the model HAS seen: {cin / len(lname_counter)}")
print(f"Percent of names mentioned that the model has NOT seen {cout / len(non_names_counter)}")
