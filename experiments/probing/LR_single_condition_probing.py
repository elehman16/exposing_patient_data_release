import argparse
from typing import Dict, List, Set

import numpy as np
from experiments.metrics import precision_at_k
from experiments.probing.common import generate_name_condition_template, get_cls_embeddings
from experiments.utilities import (
    PatientInfo,
    filter_condition_code_by_count,
    get_condition_code_to_count,
    get_condition_code_to_descriptions,
    get_subject_id_to_patient_info,
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


def get_frequency_bins(condition_code_to_count: Dict[str, int], condition_type: str) -> List[List[str]]:
    """Find which conditions belong in what bins.
    @param condition_code_to_count is a dictionary of conditions to # of occurances
    @param condition_type is in [icd9, medcat]
    @return a 2D array, where each inner array represents a different frequency
    bin. Each inner array contains condition codes belonging to that bin.
    """
    nbins = 5
    if condition_type == "medcat":
        max_count = max(condition_code_to_count.values())
        bin_cut_offs = [max_count / nbins * i + 1 for i in range(1, nbins + 1)]
    elif condition_type == "icd9":
        bin_cut_offs = [3, 5, 10, 20, 10000]

    bins = [[] for _ in range(nbins)]
    for code, count in condition_code_to_count.items():
        for n in range(nbins):
            if count <= bin_cut_offs[n]:
                bins[n].append(code)
                break

    return bins


def get_non_zero_count_conditions(
    set_to_use: List[str], subject_ids: List[str], subject_id_to_patient_info: Dict[str, PatientInfo]
) -> Set[str]:
    condition_code_to_count = {condition: 0 for condition in set_to_use}
    for subject_id in subject_ids:
        for condition in subject_id_to_patient_info[subject_id].CONDITIONS:
            if condition in condition_code_to_count:
                condition_code_to_count[condition] += 1

    return set([code for code, count in condition_code_to_count.items() if count > 0])


def train_and_evaluate(
    model: BertModel,
    tokenizer: BertTokenizer,
    condition_type: str,
    sampling_bin: int,
    n: int,
    metrics_output_path: str,
):
    """Train and evaluate the model on N conditions.
    @param n is the number of conditions to sample the bin from.
    """
    ### Get Relevant Data

    subject_id_to_patient_info = get_subject_id_to_patient_info(condition_type=condition_type)
    condition_code_to_count = get_condition_code_to_count(condition_type=condition_type)
    condition_code_to_description = get_condition_code_to_descriptions(condition_type=condition_type)

    set_to_use = filter_condition_code_by_count(condition_code_to_count, min_count=0, max_count=500000)

    binned_conditions = get_frequency_bins(condition_code_to_count, condition_type)

    subject_ids = sorted(list(subject_id_to_patient_info.keys()))
    train_subject_ids, test_subject_ids = train_test_split(
        subject_ids, train_size=0.5, random_state=2021, shuffle=True
    )

    ### Filter condition in each bin so we have atleast one positive training example
    ### And One positive test example
    ### Otherwise, we can't train a LR model or calculate roc_auc_score

    train_set_conditions = get_non_zero_count_conditions(
        set_to_use, train_subject_ids, subject_id_to_patient_info
    )
    test_set_conditions = get_non_zero_count_conditions(
        set_to_use, test_subject_ids, subject_id_to_patient_info
    )
    binned_conditions = [set(bin_) & train_set_conditions & test_set_conditions for bin_ in binned_conditions]
    binned_conditions = [sorted(list(bin_)) for bin_ in binned_conditions]

    ### Sample condition in selected bin

    condition_bin = binned_conditions[sampling_bin]
    np.random.seed(2021)
    sampled_conditions = np.random.choice(condition_bin, size=n, replace=False)

    ## Train a Classifier for Each Condition

    auc_score_list, precision_at_10_list = [], []
    for condition in tqdm(sampled_conditions):
        desc = condition_code_to_description[condition]

        ## Get all train templates and labels for all train patients, for this condition
        train_templates = []
        train_labels = []
        for subject_id in train_subject_ids:
            patient_info = subject_id_to_patient_info[subject_id]
            template = generate_name_condition_template(
                patient_info.FIRST_NAME, patient_info.LAST_NAME, patient_info.GENDER, desc
            )
            label = condition in patient_info.CONDITIONS

            train_templates.append(template)
            train_labels.append(label)

        ## Resample to Upsample positive examples

        negative_indices = [i for i, x in enumerate(train_labels) if x == 0]
        positive_indices = [i for i, x in enumerate(train_labels) if x == 1]

        # We set replace = to False in another file; does this matter?
        positive_indices = resample(
            positive_indices, replace=True, n_samples=len(negative_indices), random_state=2021
        )
        total_indices = negative_indices + positive_indices

        train_templates = [train_templates[i] for i in total_indices]
        train_labels = [train_labels[i] for i in total_indices]

        ## Train the LR model

        train_embeddings = get_cls_embeddings(model, tokenizer, train_templates, disable_tqdm=True)
        clf = MLPClassifier(hidden_layer_sizes=(128,), random_state=2021).fit(train_embeddings, train_labels)

        ## Get all test templates and labels for all test patients, for this condition

        test_templates = []
        test_labels = []

        for subject_id in test_subject_ids:
            patient_info = subject_id_to_patient_info[subject_id]
            template = generate_name_condition_template(
                patient_info.FIRST_NAME, patient_info.LAST_NAME, patient_info.GENDER, desc
            )
            label = condition in patient_info.CONDITIONS

            test_templates.append(template)
            test_labels.append(label)

        ## Get Embeddings for all test patients, and make prediction with LR model

        test_embeddings = get_cls_embeddings(model, tokenizer, test_templates, disable_tqdm=True)
        test_predictions = clf.predict_proba(test_embeddings)[:, 1]

        auc_score = roc_auc_score(test_labels, test_predictions)
        precision_at_10 = precision_at_k(test_labels, test_predictions, k=10)

        auc_score_list.append(auc_score)
        precision_at_10_list.append(precision_at_10)

    from experiments.MLM.common import mean_std_as_string

    with open(f"{metrics_output_path}/results.txt", "w") as f:
        f.write(mean_std_as_string("Model AUC", auc_score_list))
        f.write(mean_std_as_string("Model P@K", precision_at_10_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition-type", help="Are we using Stanza conditions?", choices=["icd9", "medcat"]
    )
    parser.add_argument("--model", help="Location of the model", type=str, required=True)
    parser.add_argument("--tokenizer", help="Location of the tokenizer", type=str, required=True)
    parser.add_argument("--conditions", help="Number of conditions to test per bin", type=int, default=50)
    parser.add_argument(
        "--frequency-bin",
        help="Which frequency bin to use?",
        type=int,
    )
    parser.add_argument("--metrics-output-path", type=str)
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    model = BertModel.from_pretrained(args.model).cuda().eval()

    import os

    metrics_output_path = args.metrics_output_path if args.metrics_output_path is not None else args.model
    metrics_output_path = os.path.join(
        metrics_output_path,
        f"LR_single_conditions_probing/{args.condition_type}_{args.conditions}_{args.frequency_bin}",
    )
    os.makedirs(metrics_output_path, exist_ok=True)

    train_and_evaluate(
        model, tokenizer, args.condition_type, args.frequency_bin, args.conditions, metrics_output_path
    )
