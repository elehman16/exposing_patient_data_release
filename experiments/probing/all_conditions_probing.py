import argparse
from typing import Dict, List

import numpy as np
import torch
from experiments.metrics import precision_at_k
from experiments.utilities import (
    filter_condition_code_by_count,
    get_condition_code_to_count,
    get_condition_code_to_descriptions,
    get_condition_labels_as_vector,
    get_subject_id_to_patient_info,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast


def generate_condition_only_template(condition_description: str):
    return f"[CLS] {condition_description.strip()} [SEP]"


def generate_name_condition_template(
    first_name: str, last_name: str, gender: str, condition_description: str
):
    title = "Mr" if gender == "M" else "Mrs"  # I guess just assume married w/e idk ?
    return f"[CLS] {title} {first_name} {last_name} is a yo patient with {condition_description} [SEP]"


def get_cls_embeddings(model, tokenizer, templates: List[str], disable_tqdm: bool = False) -> np.ndarray:
    embeddings = []
    batch_size = 500
    for b in tqdm(range(0, len(templates), batch_size), disable=disable_tqdm):
        batch = templates[b : b + batch_size]
        split_texts = [template.split() for template in batch]
        batch = tokenizer(
            text=split_texts,
            is_split_into_words=True,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        with torch.no_grad():
            predictions = model(batch.input_ids.cuda(), attention_mask=batch.attention_mask.cuda())
            cls_embeddings = predictions.pooler_output.cpu().data.numpy()
            embeddings.append(cls_embeddings)

    return np.concatenate(embeddings, axis=0)


def train_and_evaluate(model, tokenizer, condition_type: str, mode: str, prober: str):
    """Train and evaluate the model trained on the data.

    Args:
        model: is the BERT model used to encode the CLS tokens.
        tokenizer: is the BERT tokenizer.
        condition_type: is whether or not we use stanza labels.
        mode: is what binary element we are trying to predict via the template.
        f: is the file location of where to save the model.
    """

    ### Get Relevant Data

    subject_id_to_patient_info = get_subject_id_to_patient_info(condition_type=condition_type)
    condition_code_to_count = get_condition_code_to_count(condition_type=condition_type)
    condition_code_to_description = get_condition_code_to_descriptions(condition_type=condition_type)

    if condition_type == "stanza":
        set_to_use = filter_condition_code_by_count(condition_code_to_count, min_count=50, max_count=500000)
    elif condition_type == "icd9":
        set_to_use = filter_condition_code_by_count(condition_code_to_count, min_count=0, max_count=500000)
    else:
        raise NotImplementedError()

    condition_code_to_index: Dict[str, int] = dict(zip(set_to_use, range(len(set_to_use))))

    ## Divide patients into train and test group

    all_subject_ids = sorted(list(subject_id_to_patient_info.keys()))
    train_subject_ids, test_subject_ids = train_test_split(
        all_subject_ids, random_state=2021, test_size=0.5, shuffle=True
    )

    ## Get training example by generating template for all train patients and all conditions

    subject_condition_templates = []
    subject_condition_labels = []

    for subject_id in train_subject_ids:
        patient_info = subject_id_to_patient_info[subject_id]
        for condition in set_to_use:
            desc = condition_code_to_description[condition]
            if mode == "name_and_condition":
                template = generate_name_condition_template(
                    patient_info.FIRST_NAME, patient_info.LAST_NAME, patient_info.GENDER, desc
                )
            elif mode == "condition_only":
                template = generate_condition_only_template(desc)
            else:
                raise NotImplementedError(f"{mode} is not available")
            subject_condition_templates.append(template)

        condition_labels = get_condition_labels_as_vector(patient_info.CONDITIONS, condition_code_to_index)
        subject_condition_labels += list(condition_labels)

    ## Downsample negative labels since most patients only have few positive conditions

    negative_indices = [i for i, x in enumerate(subject_condition_labels) if x == 0]
    positive_indices = [i for i, x in enumerate(subject_condition_labels) if x == 1]

    negative_indices = resample(
        negative_indices, replace=False, n_samples=len(positive_indices), random_state=2021
    )
    total_indices = negative_indices + positive_indices

    train_templates = [subject_condition_templates[i] for i in total_indices]
    train_labels = [subject_condition_labels[i] for i in total_indices]

    print(len(train_templates))

    ## Get [CLS] token embedding for each template and train a LR classifier

    train_cls_embeddings = get_cls_embeddings(model, tokenizer, train_templates)

    print(f"Training {prober} Model")
    if prober == "LR" :
        classifier = LogisticRegression(random_state=2021, max_iter=10000).fit(
            train_cls_embeddings, train_labels
        )
    elif prober == "MLP" :
        classifier = MLPClassifier(hidden_layer_sizes=(128,), random_state=2021).fit(train_cls_embeddings, train_labels)
    else :
        raise NotImplementedError(f"{prober} not implemented")
    print(f"{prober} Model Trained")

    ## Get templates and labels for test set patients

    auc_scores, paks = [], []

    for subject_id in tqdm(test_subject_ids):
        test_templates = []
        patient_info = subject_id_to_patient_info[subject_id]
        for condition in set_to_use:
            desc = condition_code_to_description[condition]
            if mode == "name_and_condition":
                template = generate_name_condition_template(
                    patient_info.FIRST_NAME, patient_info.LAST_NAME, patient_info.GENDER, desc
                )
            elif mode == "condition_only":
                template = generate_condition_only_template(desc)
            else:
                raise NotImplementedError(f"{mode} is not available")
            test_templates.append(template)

        condition_labels = get_condition_labels_as_vector(patient_info.CONDITIONS, condition_code_to_index)
        test_labels = list(condition_labels)

        test_cls_embeddings = get_cls_embeddings(model, tokenizer, test_templates, disable_tqdm=True)
        test_predictions = classifier.predict_proba(test_cls_embeddings)[:, 1]

        model_auc = roc_auc_score(test_labels, test_predictions)
        model_precision_at_k = precision_at_k(test_labels, test_predictions, k=10)

        auc_scores.append(model_auc)
        paks.append(model_precision_at_k)

    print("Average P@10: {}".format(np.average(paks)))
    print("Average AUC: {}".format(np.average(auc_scores)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-type", type=str, choices=["icd9", "stanza"])
    parser.add_argument("--model", help="Location of the model", type=str)
    parser.add_argument("--tokenizer", help="Location of the tokenizer", type=str)
    parser.add_argument(
        "--mode",
        help="Normal, only use masked conditons for predictions (no names)?",
        type=str,
        choices=["name_and_condition", "condition_only"],
    )
    parser.add_argument("--prober", type=str, choices=["LR", "MLP"])
    args = parser.parse_args()

    # Load pre-trained model tokenizer (vocabulary)
    # '/home/eric/dis_rep/nyu_clincalBERT/clinicalBERT/notebook/bert_uncased/'
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)

    # Load pre-trained model (weights)
    # '/home/eric/dis_rep/nyu_clincalBERT/convert_to_pytorch/all_useful_100k/'
    model = BertModel.from_pretrained(args.model).cuda().eval()
    train_and_evaluate(model, tokenizer, args.condition_type, args.mode, args.prober)


if __name__ == "__main__":
    main()
