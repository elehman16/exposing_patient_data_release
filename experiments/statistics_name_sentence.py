import pandas as pd

nf = '/home/eric/exposing_patient_data_release/model_outputs/ClinicalBERT_1a/input_data/SUBJECT_ID_to_NOTES_1a.csv.sentences'
sif = '/home/eric/exposing_patient_data_release/setup_outputs/SUBJECT_ID_to_NAME.csv'

# load names 
with open(nf) as tmp:
    txt_ln = tmp.read().split('\n')

si_to_n = pd.read_csv(sif)
fn = list(set(si_to_n.FIRST_NAME.values))
ln = list(si_to_n.LAST_NAME.values)
map_ = {'either': 0, 'first': 0, 'last': 0}
for s in txt_ln:
    either = False
    s = s.split(' ')
    for f in fn:
        if f.lower() in s:
            map_['first'] += 1
            either = True
            break

    for l in ln:
        if l.lower() in s:
            map_['last'] += 1
            either = True
            break

    if either:
        map_['either'] += 1


import pdb; pdb.set_trace()

