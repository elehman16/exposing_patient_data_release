import argparse
from typing import Dict

import numpy as np
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

from experiments.probing.common import (
    generate_condition_only_template,
    generate_name_condition_template,
    get_cls_embeddings,
)


def run_probe(
    model: BertModel,
    tokenizer: BertTokenizerFast,
    condition_type: str,
    template_mode: str,
    prober: str,
    metrics_output_path: str,
):
    """Train and evaluate the model trained on the data.

    Args:
        condition_type: icd9 or Stanza
        template_mode: Choices in [name_and_condition, condition_only].
                        Specify if name should be included in template
        prober: LR or MLP
    """

    ### Get Relevant Data

    subject_id_to_patient_info = get_subject_id_to_patient_info(condition_type=condition_type)
    condition_code_to_count = get_condition_code_to_count(condition_type=condition_type)
    condition_code_to_description = get_condition_code_to_descriptions(condition_type=condition_type)

    set_to_use = filter_condition_code_by_count(condition_code_to_count, min_count=0, max_count=500000)

    condition_code_to_index: Dict[str, int] = dict(zip(set_to_use, range(len(set_to_use))))

    ## Divide patients into train and test group

    all_subject_ids = sorted(list(subject_id_to_patient_info.keys()))

    ## Sample 10K subjects because 27K takes timeeeeeee 
    all_subject_ids = sorted(resample(all_subject_ids, replace=False, n_samples=10000, random_state=2021))

    train_subject_ids, test_subject_ids = train_test_split(
        all_subject_ids, random_state=2021, test_size=0.5, shuffle=True
    )

    print(f"Train Subject Ids : {len(train_subject_ids)}")
    print(f"Test Subject Ids : {len(test_subject_ids)}")
    import pdb; pdb.set_trace()

    ## Get training example by generating template for all train patients and all conditions

    subject_condition_templates = []
    subject_condition_labels = []

    for subject_id in train_subject_ids:
        patient_info = subject_id_to_patient_info[subject_id]
        for condition in set_to_use:
            desc = condition_code_to_description[condition]
            if template_mode == "name_and_condition":
                template = generate_name_condition_template(
                    patient_info.FIRST_NAME, patient_info.LAST_NAME, patient_info.GENDER, desc
                )
            elif template_mode == "condition_only":
                template = generate_condition_only_template(desc)
            else:
                raise NotImplementedError(f"{template_mode} is not available")
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
    if prober == "LR":
        classifier = LogisticRegression(random_state=2021, max_iter=10000).fit(
            train_cls_embeddings, train_labels
        )
    elif prober == "MLP":
        classifier = MLPClassifier(hidden_layer_sizes=(128,), random_state=2021).fit(
            train_cls_embeddings, train_labels
        )
    else:
        raise NotImplementedError(f"{prober} not implemented")
    print(f"{prober} Model Trained")

    ## Get templates and labels for test set patients

    auc_scores, paks = [], []

    for subject_id in tqdm(test_subject_ids):
        test_templates = []
        patient_info = subject_id_to_patient_info[subject_id]
        for condition in set_to_use:
            desc = condition_code_to_description[condition]
            if template_mode == "name_and_condition":
                template = generate_name_condition_template(
                    patient_info.FIRST_NAME, patient_info.LAST_NAME, patient_info.GENDER, desc
                )
            elif template_mode == "condition_only":
                template = generate_condition_only_template(desc)
            else:
                raise NotImplementedError(f"{template_mode} is not available")
            test_templates.append(template)

        condition_labels = get_condition_labels_as_vector(patient_info.CONDITIONS, condition_code_to_index)
        test_labels = list(condition_labels)

        test_cls_embeddings = get_cls_embeddings(model, tokenizer, test_templates, disable_tqdm=True)
        test_predictions = classifier.predict_proba(test_cls_embeddings)[:, 1]

        try:
            model_auc = roc_auc_score(test_labels, test_predictions)
            model_precision_at_k = precision_at_k(test_labels, test_predictions, k=10)

            auc_scores.append(model_auc)
            paks.append(model_precision_at_k)
        except:
            continue

    from experiments.MLM.common import mean_std_as_string

    with open(f"{metrics_output_path}/results.txt", "w") as f:
        f.write(mean_std_as_string("Model AUC", auc_scores))
        f.write(mean_std_as_string("Model P@K", paks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Location of the model", type=str, required=True)
    parser.add_argument("--tokenizer", help="Location of the tokenizer", type=str, required=True)
    parser.add_argument("--condition-type", type=str, choices=["icd9", "medcat"], required=True)
    parser.add_argument(
        "--template-mode",
        help="Normal, only use masked conditons for predictions (no names)?",
        type=str,
        choices=["name_and_condition", "condition_only"],
        required=True,
    )
    parser.add_argument(
        "--prober",
        type=str,
        choices=["LR", "MLP"],
        required=True,
        help="Which probing model to train on top of BERT embeddings ?",
    )
    parser.add_argument("--metrics-output-path", type=str)
    args = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)
    model = BertModel.from_pretrained(args.model).cuda().eval()

    import os

    metrics_output_path = args.metrics_output_path if args.metrics_output_path is not None else args.model
    metrics_output_path = os.path.join(
        metrics_output_path,
        f"all_conditions_probing/{args.condition_type}_{args.template_mode}_{args.prober}",
    )
    os.makedirs(metrics_output_path, exist_ok=True)

    run_probe(model, tokenizer, args.condition_type, args.template_mode, args.prober, metrics_output_path)
