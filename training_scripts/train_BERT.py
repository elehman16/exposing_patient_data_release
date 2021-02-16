from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm 

tqdm.pandas()

import re
import string
import os
import subprocess

parser = ArgumentParser()
parser.add_argument("--input-file", required=True)
parser.add_argument("--output-dir", required=True)


def string_cleanup(x):
    y = re.sub("\\[(.*?)\\]", "", x)  # remove de-identified brackets
    # remove 1.2. since the segmenter segments based on this
    y = re.sub("[0-9]+\.", "", y)
    y = re.sub("dr\.", "doctor", y)
    y = re.sub("m\.d\.", "md", y)
    y = re.sub("admission date:", "", y)
    y = re.sub("discharge date:", "", y)
    y = re.sub("--|__|==", "", y)

    # remove, digits, spaces
    y = y.translate(str.maketrans("", "", string.digits))
    y = re.sub(r"[ \t\r\f\v]+", " ", y)
    return y


def preprocessing(df_notes):
    df_notes["TEXT"] = df_notes["TEXT"].str.replace("\r", " ")
    df_notes["TEXT"] = df_notes["TEXT"].apply(str.strip)
    df_notes["TEXT"] = df_notes["TEXT"].str.lower()

    df_notes["TEXT"] = df_notes["TEXT"].progress_apply(lambda x: string_cleanup(x))

    return df_notes


def run(input_file, output_dir):
    df = pd.read_csv(input_file)
    df = preprocessing(df)

    data_folder = os.path.join(output_dir, "input_data")
    tmp_file_for_sentences = f"{data_folder}/{os.path.basename(input_file)}.sentences"
    os.makedirs(os.path.dirname(tmp_file_for_sentences), exist_ok=True)

    with open(tmp_file_for_sentences, "w") as tmp_file:
        for sentences in df.TEXT.values:
            if len(sentences) > 0:
                tmp_file.write(sentences.strip() + "\n")

    """
    Training Code for BERT
    """

    subprocess.run(
        [
            "bash", "training_scripts/train_bert.sh",
        ],
        check=True,
        env={
            "MODEL_OUTPUT_FOLDER": output_dir,
            "PRETRAINING_TXT_FILE": tmp_file_for_sentences,
            **os.environ
        },
    )


if __name__ == "__main__":
    """
    Example Usage:
        python train_BERT.py \
            --input-file SUBJECT_ID_to_NOTES_{}.csv \
            --output-dir model_outputs/ClinicalBERT_{}/

    """
    args = parser.parse_args()
    run(args.input_file, args.output_dir)
