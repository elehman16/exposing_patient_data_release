import os
from argparse import ArgumentParser
import pandas as pd

CATEGORIES_TO_USE = ["Physician ", "Nursing", "Nursing/other", "Discharge summary"]


def run(input_file, output_file):
    notes_df = pd.read_csv(input_file)
    print(f"Loaded Data from CSV -- {len(notes_df)}")

    notes_df = notes_df[notes_df.CATEGORY.isin(CATEGORIES_TO_USE)]
    print(f"Filtered to -- {len(notes_df)}")
    print(notes_df.CATEGORY.unique())

    notes_df = notes_df[~notes_df.TEXT.isna()]
    notes_df = notes_df[notes_df.TEXT.apply(lambda x: len(x.strip()) > 0)]
    notes_df = notes_df[["SUBJECT_ID", "TEXT"]].sort_values(by="SUBJECT_ID")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    notes_df.to_csv(output_file, index=False)


parser = ArgumentParser()
parser.add_argument("--input-file", required=True)
parser.add_argument("--output-file", required=True)


if __name__ == "__main__":
    """
    This file takes in the original NOTEEVENTS.csv and
    1. Subsets to four classes kept
    2. Remove any empty / na strings
    3. Sentencize the notes and then restructure them as \n.join[sentences]
    Usage:
        - python subject_id_to_name.py --input-file NOTEEVENTS.csv --output-file SUBJECT_ID_to_NOTES_original.csv

    Output Format:
        SUBJECT_ID,TEXT
        249,Note 1
        249,Note 2
        249,Note 3
        250,Note 1
        250,Note 2
    """

    args = parser.parse_args()
    run(args.input_file, args.output_file)
