import os
from argparse import ArgumentParser

import pandas as pd
import stanza
from tqdm import tqdm

nlp = stanza.Pipeline('en', package='mimic', processors={
                      'ner': 'i2b2'}, use_gpu=False)

tqdm.pandas()

def find_stanza_conditions(text):
    conditions = []
    for sentence in text.split("\n"):
        doc = nlp(sentence)
        for ent in doc.entities:
            if ent.type == 'PROBLEM':
                conditions.append(ent.text)

    return list(set(conditions))


def run(input_file, output_file):
    notes_df = pd.read_csv(input_file)
    print(f"Loaded Data from CSV -- {len(notes_df)}")

    notes_df["STANZA"] = notes_df["TEXT"].progress_apply(find_stanza_conditions)

    notes_df = notes_df[["SUBJECT_ID", "STANZA"]].sort_values(by="SUBJECT_ID")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    notes_df.to_csv(output_file, index=False)


parser = ArgumentParser()
parser.add_argument("--input-file", required=True)
parser.add_argument("--output-file", required=True)


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
    run(args.input_file, args.output_file)
