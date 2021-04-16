import os, json 
import pandas as pd
from tqdm import tqdm

from medcat.cdb import CDB 
cdb = CDB()
cdb.load_dict(os.environ["MEDCAT_CDB_FILE"]) 
print("Loaded CDB")

def denormalize(entities) :
    if len(entities) == 0 :
        return [], None
    cuis = [cui for text, cui, icd10 in entities]
    cuis = list(set(cuis))

    overlaps = 0 
    for text, cui, icd10 in entities:
        cui_name = cdb.cui2pretty_name[cui].replace(" ", "").lower()
        text = text.replace(" ", "").lower()
        abbr = "".join([c[0] for c in cdb.cui2pretty_name[cui].split()]).lower()
        if text == cui_name or text in cui_name or cui_name in text or abbr in text:
            overlaps += 1

    return cuis, overlaps / len(entities)

switch_comma = lambda x : x.split(",", 1)[1].strip() + " " + x.split(",", 1)[0].strip()

def run(input_file: str, output_file: str, output_descriptions_file: str) :
    data = [json.loads(line) for line in open(input_file)]
    subject_ids, conditions = [], []
    overlaps = []
    for d in tqdm(data) :
        cuis, overlap = denormalize(d["CONDITIONS"])
        subject_ids += [d["SUBJECT_ID"] for _ in range(len(cuis))]
        conditions += cuis
        overlaps.append(overlap)

    output_df = pd.DataFrame({"SUBJECT_ID": subject_ids, "CODE": conditions})
    output_df.sort_values(by=["SUBJECT_ID", "CODE"]).to_csv(output_file, index=False)

    cuis = sorted(list(set(conditions)))
    code_descriptions = [cdb.cui2pretty_name[cui] for cui in cuis]
    code_descriptions = pd.DataFrame({"CODE": cuis, "DESCRIPTION": code_descriptions})
    code_descriptions["DESCRIPTION"] = code_descriptions.DESCRIPTION.apply(lambda x : switch_comma(x) if "," in x else x)
    code_descriptions.to_csv(output_descriptions_file, index=False)

from argparse import ArgumentParser

if __name__ == "__main__" :
    parser = ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--output-descriptions-file", required=True)

    args = parser.parse_args()

    run(args.input_file, args.output_file, args.output_descriptions_file)
    
