import os
import re
from argparse import ArgumentParser

import pandas as pd
import swifter
from tqdm import tqdm

tqdm.pandas()

from spacy.lang.en import English

sentence_nlp = English()  # just the language with no model
sentence_nlp.add_pipe(sentence_nlp.create_pipe("sentencizer"))
sentence_nlp.max_length = 2000000


# nlp praser may not work when there is only one token. In these cases, we just remove them as note that has length 1 usually is some random stuff


def convert_to_sentence(text):
    doc = sentence_nlp(text)
    text = []
    try:
        for sent in doc.sents:
            st = str(sent).strip()
            if len(st) < 20:
                # a lot of abbreviation is segmented as one line. But these are all describing the previous things
                # so I attached it to the sentence before
                if len(text) != 0:
                    text[-1] = " ".join((text[-1], st))
                else:
                    text = [st]
            else:
                text.append(re.sub(r"\s+", " ", st.replace("\n", " ").strip()))
    except:
        print(doc)

    return "\n".join([sent for sent in text if len(sent) > 0])


CATEGORIES_TO_USE = ["Physician ", "Nursing", "Nursing/other", "Discharge summary"]


def run(input_file, output_file):
    notes_df = pd.read_csv(input_file)
    print(f"Loaded Data from CSV -- {len(notes_df)}")

    notes_df = notes_df[notes_df.CATEGORY.isin(CATEGORIES_TO_USE)]
    print(f"Filtered to -- {len(notes_df)}")
    print(notes_df.CATEGORY.unique())

    notes_df["TEXT"] = (
        notes_df["TEXT"].swifter.progress_bar(enable=True).allow_dask_on_strings().apply(convert_to_sentence)
    )

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
