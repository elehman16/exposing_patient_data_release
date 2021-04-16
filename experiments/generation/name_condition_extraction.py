import glob

import config
import spacy
import pandas as pd
from tqdm import tqdm

from joblib import Parallel, delayed

from setup_scripts.subject_id_to_medcat_preprocess import get_entities
from experiments.utilities import get_subject_id_to_patient_info

nlp = spacy.load("en")

def has_name(sentences, names):
    sentences_with_name = []
    for text in sentences :
        text = text.replace("[CLS]", "").replace("[SEP]", "").strip()
        doc = nlp(text)

        name_tokens = set([token.text.lower() for token in doc if token.ent_type_ == "PERSON"])

        if len(name_tokens & names) > 0 :
            sentences_with_name.append((text, name_tokens & names))

    return sentences_with_name
        
    
def preprocess_parallel(function, texts, chunksize=100, **kwargs):
    chunker = (texts[i : i + chunksize] for i in range(0, len(texts), chunksize))
    executor = Parallel(n_jobs=28, backend="multiprocessing", prefer="processes", verbose=20)
    do = delayed(function)
    tasks = (do(chunk, **kwargs) for chunk in chunker)
    result = executor(tasks)
    return [text for chunk in result for text in chunk]


from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--sample-files")
parser.add_argument("--metrics-output-path")


if __name__ == "__main__":
    args = parser.parse_args()

    metrics_output_path = args.metrics_output_path
    print(f"Saving results to {metrics_output_path}")

    txt_files = glob.glob(args.sample_files)

    all_sentences = []

    for f in tqdm(txt_files):
        with open(f) as tmp:
            all_sentences += [line.strip() for line in tmp]

    df = pd.read_csv(config.SUBJECT_ID_to_NAME)
    modified = set(pd.read_csv(config.MODIFIED_SUBJECT_IDS)["SUBJECT_ID"])
    df = df[df["SUBJECT_ID"].isin(modified)]

    # Lower case the first and last names
    df["FIRST_NAME"] = df["FIRST_NAME"].apply(lambda x: str(x).lower())
    df["LAST_NAME"] = df["LAST_NAME"].apply(lambda x: str(x).lower())

    first_names = set(df["FIRST_NAME"].values)
    last_names = set(df["LAST_NAME"].values)

    sample_with_names = preprocess_parallel(has_name, all_sentences, chunksize=1000, names=first_names | last_names)

    print(len(sample_with_names), len(all_sentences))

    subject_id_to_patient_info = get_subject_id_to_patient_info("medcat")

    name_to_subject_id = {}
    for subject_id, patient_info in subject_id_to_patient_info.items() :
        name_to_subject_id.setdefault(patient_info.FIRST_NAME.lower(), []).append(subject_id)
        name_to_subject_id.setdefault(patient_info.LAST_NAME.lower(), []).append(subject_id)

    found_sentences = 0
    for sample_sentence, sample_names in tqdm(sample_with_names) :
        entities = [x for x in get_entities(sample_sentence)]
        subject_ids = list(set([subject_id for name in sample_names if name in name_to_subject_id for subject_id in name_to_subject_id[name] ]))
        cuis = set([condition for sid in subject_ids for condition in subject_id_to_patient_info[sid].CONDITIONS])

        found_entities = [ent for ent in entities if ent[1] in cuis]
        found_sentences += 1 if (len(found_entities) > 0) else 0 

    metric = found_sentences / (1e-7 + len(sample_with_names))
    print(f"{metric}")

    with open(f"{metrics_output_path}/sampling_results_condition.txt", "w") as f:
        f.write(f"{metric}\n")
