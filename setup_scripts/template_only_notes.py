from experiments.utilities import get_subject_id_to_patient_info, get_condition_code_to_descriptions

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--condition-type", required=True)
parser.add_argument("--output-file", required=True)

args = parser.parse_args()


def generate_name_condition_template(
    first_name: str, last_name: str, gender: str, condition_description: str
):
    title = "Mr" if gender == "M" else "Mrs"  # I guess just assume married w/e idk ?
    return f"{title} {first_name} {last_name} is a yo patient with {condition_description}"


data = get_subject_id_to_patient_info(args.condition_type)
desc = get_condition_code_to_descriptions(args.condition_type)

templates = [
    "\n".join(
        [generate_name_condition_template(d.FIRST_NAME, d.LAST_NAME, d.GENDER, desc[c]) for c in d.CONDITIONS]
    )
    for d in data.values()
]
subject_ids = [d for d in data.keys()]

import pandas as pd

df = pd.DataFrame({"SUBJECT_ID": subject_ids, "TEXT": templates})

df.to_csv(args.output_file, index=False)
