from argparse import ArgumentParser
from typing import Dict

import config
import pandas as pd


def icd_to_english() -> Dict[str, str]:
    """
    Convert the ICD-9 Codes to short text descriptions.
    @return a mapping of ICD codes to short descriptions (dictionary, str -> str).
    """
    code_to_english = {}
    with open(config.ICD_9_CONVERSION) as tmp:
        text = tmp.read().split("\n")[:-1]

    for row in text:
        code = row.split(" ")[0]
        description = row[len(code) :]
        code_to_english[code] = description.strip()

    return code_to_english


def run(input_file: str, output_file: str, output_descriptions_file: str):
    patients_icd_df = pd.read_csv(input_file)
    code_to_english = icd_to_english()

    valid_codes = set(list(code_to_english.keys()))

    ## Filter to codes in code description dict
    patients_icd_df = patients_icd_df[patients_icd_df.ICD9_CODE.isin(valid_codes)]
    patients_icd_df = patients_icd_df.rename(columns={"ICD9_CODE": "CODE"})[["SUBJECT_ID", "CODE"]]

    patients_icd_df.sort_values(by=["SUBJECT_ID", "CODE"]).to_csv(output_file, index=False)

    codes, descriptions = list(zip(*code_to_english.items()))
    descriptions_df = pd.DataFrame({"CODE": codes, "DESCRIPTION": descriptions})
    descriptions_df.to_csv(output_descriptions_file, index=False)


if __name__ == "__main__":
    """
    Usage:
        - python subject_id_to_icd9.py \
            --input-file DIAGNOSES_ICD.csv \
            --output-file SUBJECT_ID_to_ICD9.csv \
            --output-descriptions-file ICD9_Descriptions.csv

    Output Format:
        SUBJECT_ID,CODE
        109,"40301"
        109,"486"
        109,"58281"
    """
    parser = ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--output-descriptions-file", required=True)
    args = parser.parse_args()
    run(args.input_file, args.output_file, args.output_descriptions_file)
