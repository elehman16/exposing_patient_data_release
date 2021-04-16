from argparse import ArgumentParser

import pandas as pd
import os
import numpy as np
import subprocess


def run(input_file, output_dir, distributed, n_jobs, job_num):
    df = pd.read_csv(input_file)

    if distributed:
        df = np.array_split(df, n_jobs)[job_num]

    print(f"Loaded {len(df)} records -- {df.index.min()}-{df.index.max()}")

    data_folder = os.path.join(output_dir, "input_data")
    tmp_file_for_sentences = f"{data_folder}/{os.path.basename(input_file)}.sentences"

    if distributed :
        tmp_file_for_sentences = tmp_file_for_sentences + f".{job_num}-{n_jobs}"

    if not(os.path.exists(tmp_file_for_sentences)):
        os.makedirs(os.path.dirname(tmp_file_for_sentences), exist_ok=True)

        with open(tmp_file_for_sentences, "w") as tmp_file:
            for sentences in df.TEXT.values:
                if len(sentences) > 0:
                    tmp_file.write(sentences.strip() + "\n")
                tmp_file.write("\n")

    """
    Training Code for BERT
    """

    subprocess.run(
        [
            "bash", "training_scripts/create_BERT_tfrecords.sh",
        ],
        check=True,
        env={
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

    parser = ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--job-num", type=int)
    args = parser.parse_args()

    run(args.input_file, args.output_dir, args.distributed, args.n_jobs, args.job_num)
