import os 
import sys 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import glob
import config
import pandas as pd 
import re

name_of_interest = ''
loc = '/home/eric/dis_rep/physionet/files/mimiciii/physionet.org/files/deid/1.1/lists/last_names_popular.txt'

def get_all_relevant_conditions(name):
    pass

def find_all(s, str_):
    return [m.start() for m in re.finditer(s, str_)]

def create_counts(l):
    dic_ = {}
    for w in l:
        if w in dic_:
            dic_[w] += 1
        else: 
            dic_[w] = 1

    return dic_ 

df = pd.read_csv(config.DF)
names = list(set(df[df['CHANGED']]['NAMES']))
names = [eval(x) for x in names]
non_names = list(set(df['NAMES']) - set(df[df['CHANGED']]['NAMES']))
non_names = [eval(x)[1].lower() for x in non_names]

non_names_counter = dict.fromkeys(non_names, 0)
lname_counter = dict.fromkeys([x[1].lower() for x in names], 0)
fname_counter = dict.fromkeys([x[0].lower() for x in names], 0)
total_name_counter = dict.fromkeys([' '.join(x).lower() for x in names], 0)

txt_files = glob.glob('extracted_data/*.txt')
for f in txt_files:
    with open(f) as tmp: txt = tmp.read()
    words = create_counts(txt.split(' '))
    bigrams = create_counts([' '.join(x) for x in zip(txt.split(" ")[:-1], txt.split(" ")[1:])])
    for n in names:
        full = ' '.join(n).lower()
        if full in bigrams: total_name_counter[full] += bigrams[full]
        if n[0].lower() in words: fname_counter[n[0].lower()] += words[n[0].lower()]
        if n[1].lower() in words: lname_counter[n[1].lower()] += words[n[1].lower()]
   
    for n in non_names:
        if n.lower() in words: 
            non_names_counter[n.lower()] += words[n.lower()]

cin = 0
for n in lname_counter.keys():
    if lname_counter[n] > 0:
        cin += 1

cout = 0
for n in non_names_counter:
    if non_names_counter[n] > 0:
        cout += 1

print(cin / len(lname_counter))
print(cout / len(non_names_counter))
import pdb; pdb.set_trace()
total_name_counter = list(zip(total_name_counter.keys(), total_name_counter.values()))
total_name_counter.sort(key=lambda x: x[-1], reverse=True)

lname_counter = list(zip(lname_counter.keys(), lname_counter.values()))
lname_counter.sort(key=lambda x: x[-1], reverse=True)

fname_counter = list(zip(fname_counter.keys(), fname_counter.values()))
fname_counter.sort(key=lambda x: x[-1], reverse=True)

