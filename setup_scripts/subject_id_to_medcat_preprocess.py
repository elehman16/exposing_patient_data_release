import os
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm
import numpy as np

from medcat.cat import CAT
from medcat.utils.vocab import Vocab
from medcat.cdb import CDB 

vocab = Vocab()
vocab.load_dict(os.environ["MEDCAT_VOCAB_FILE"])
print("Loaded Vocab")

# Load the cdb model you downloaded
cdb = CDB()
cdb.load_dict(os.environ["MEDCAT_CDB_FILE"]) 
print("Loaded CDB")

# create cat
cat = CAT(cdb=cdb, vocab=vocab)
cat.spacy_cat.TUI_FILTER = ['T047', 'T048', 'T184']

tqdm.pandas()

def get_entities(text) :
    doc = cat.get_entities(text)
    relevant_entities = []
    for ent in doc :
        if "icd10" in ent["info"] :
            ent_string = text[ent["start"]:ent['end']]
            if ent_string.lower() in ["ms", "mr", "mrs"] :
                continue
            cui = ent["cui"]
            icd_codes = tuple(sorted([x["chapter"] for x in ent["info"]["icd10"]]))
            if "R69" in icd_codes:
                continue
            relevant_entities.append((ent_string, cui, icd_codes))

    return list(set(relevant_entities))


def run(input_file, output_file, distributed: bool, args):
    notes_df = pd.read_csv(input_file).reset_index(drop=True)
    if distributed :
        notes_df = np.array_split(notes_df, args.n_jobs)[args.job_num]

    print(f"Loaded Data from CSV -- {len(notes_df)} -- {min(notes_df.index)} -- {max(notes_df.index)}")

    notes_df["CONDITIONS"] = notes_df["TEXT"].progress_apply(get_entities)

    notes_df = notes_df[["SUBJECT_ID", "CONDITIONS"]].sort_values(by="SUBJECT_ID")
    notes_df = notes_df.groupby("SUBJECT_ID")["CONDITIONS"].agg(lambda x : list(set([y for l in x for y in l]))).reset_index()

    if args.distributed :
        output_file = output_file + f".{args.n_jobs}.{args.job_num}"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    notes_df.to_json(output_file, orient="records", lines=True)


parser = ArgumentParser()
parser.add_argument("--input-file", required=True)
parser.add_argument("--output-file", required=True)
parser.add_argument("--distributed", action="store_true")
parser.add_argument("--n-jobs", type=int)
parser.add_argument("--job-num", type=int)


if __name__ == "__main__":
    '''
    Usage:
        - python subject_id_to_name.py --input-file SUBJECT_ID_to_NOTES_original.csv --output-file SUBJECT_ID_to_Stanza.csv

    Output Format:
        SUBJECT_ID,Stanza_Code
        249,Code 1
        249,Code 2
        249,Code 3
        250,Code 1
        250,Code 2
    '''

    args = parser.parse_args()
    if args.distributed :
        assert "n_jobs" in args and "job_num" in args

    run(args.input_file, args.output_file, args.distributed, args)
