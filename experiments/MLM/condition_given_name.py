import argparse
from typing import Dict, List

import numpy as np
from experiments.masked_prediction.common import (
    get_average_predicted_score,
    get_logits_from_templates,
)
from experiments.metrics import precision_at_k
from experiments.utilities import (
    filter_condition_code_by_count,
    get_condition_code_to_count,
    get_condition_code_to_descriptions,
    get_condition_counts_as_vector,
    get_condition_labels_as_vector,
    get_subject_id_to_patient_info,
)
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer


def condition_only_template(condition_length: int) -> str:
    """Generate empty template with condition replaces with [MASK] string of `condition_length`"""
    mask_string = "[MASK] " * condition_length
    return f"[CLS] {mask_string.strip()} [SEP]"


def name_with_condition_template(first_name: str, last_name: str, gender: str, condition_length: int) -> str:
    """Generate filled template with condition replaces with [MASK] string of `condition_length`"""
    title = "Mr" if gender == "M" else "Mrs"  # I guess just assume married w/e idk ?
    mask_string = "[MASK] " * condition_length
    return f"[CLS] {title} {first_name} {last_name} is a yo patient with {mask_string.strip()} [SEP]"


def evaluate(model: BertForMaskedLM, tokenizer: BertTokenizer, condition_type: str):
    """
    Evaluate the performance of the model in terms of being able to predict
    conditions associated with certain names (via masked language modelling task on templates).

    For each patient,
        For each condition,
            create template with name filled and condition masked.\ 
            get MLM probability distribution for [MASK] tokens.\ 
            get average probability of condition's wordpieces.\ 
        Compute ROC / P@10 against true condition labels.
    Report Average over patients.

    Ideally we want higher average probability for conditions that patient have, than for conditions they
    don't have. 

    We also include a condition only baseline (where template doesn't contain the patient name). The
    algorithm above remains same.

    ### Args:
        condition_type: Which conditions to load for patients. Currently take value in [icd9, stanza]
    """

    ### Load relevant data

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

    ### Get list of unique condition lengths (in wordpieces) to generate templates

    condition_code_to_wordpiece_ids: Dict[str, List[str]] = {
        condition: tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(condition_code_to_description[condition])
        )
        for condition in set_to_use
    }  ## Time Saving Measure since we only need condition wordpiece ids moving forward

    condition_wordpiece_lengths = [
        len(condition_code_to_wordpiece_ids[condition]) for condition in set_to_use
    ]
    condition_wordpiece_lengths: List[int] = sorted(
        list(set(condition_wordpiece_lengths))
    )  ## Keep Unique Lengths only

    ### Get Condition Frequency counts

    condition_baseline_counts = get_condition_counts_as_vector(
        condition_code_to_count, condition_code_to_index
    )

    ### Get Condition only template logits

    ## Generate Template for each unique condition wordpiece length
    condition_only_templates = [condition_only_template(length) for length in condition_wordpiece_lengths]

    logits = get_logits_from_templates(model, tokenizer, condition_only_templates)
    logits = {length: logit for length, logit in zip(condition_wordpiece_lengths, logits)}

    start_indices = [tokenizer.tokenize(template).index("[MASK]") for template in condition_only_templates]
    start_indices = {
        length: start_index for length, start_index in zip(condition_wordpiece_lengths, start_indices)
    }

    condition_only_logits: List[float] = []
    for condition in set_to_use:
        condition_wp_ids = condition_code_to_wordpiece_ids[condition]
        condition_length = len(condition_wp_ids)
        condition_only_logits.append(
            get_average_predicted_score(
                logits[condition_length], condition_wp_ids, start_indices[condition_length]
            )
        )

    ### Get Subject Specific Condition logits

    k = 10
    rocs = {"baseline": [], "condition_only": [], "model": []}
    precisions_at_k = {"baseline": [], "condition_only": [], "model": []}

    for subject_id, patient_info in tqdm(subject_id_to_patient_info.items()):
        condition_labels = get_condition_labels_as_vector(patient_info.CONDITIONS, condition_code_to_index)
        condition_labels = np.array(condition_labels)

        if condition_labels.sum() == 0:
            continue  ## Skip if patient is negative for all conditions

        ## Generate Template for each unique condition wordpiece length
        templates = []
        for length in condition_wordpiece_lengths:
            template = name_with_condition_template(
                patient_info.FIRST_NAME, patient_info.LAST_NAME, patient_info.GENDER, length
            )
            templates.append(template)

        ## Get logits for all templates
        logits = get_logits_from_templates(model, tokenizer, templates, normalize=False)
        logits = {length: logit for length, logit in zip(condition_wordpiece_lengths, logits)}

        ## Get Start index for (masked) condition in each template 
        start_indices = [tokenizer.tokenize(template).index("[MASK]") for template in templates]
        start_indices = {
            length: start_index for length, start_index in zip(condition_wordpiece_lengths, start_indices)
        }

        ## For each condition, get corresponding logit array and then compute average score
        condition_subject_logits = []
        for condition in set_to_use:
            condition_wp_ids = condition_code_to_wordpiece_ids[condition]
            condition_length = len(condition_wp_ids)
            condition_subject_logits.append(
                get_average_predicted_score(
                    logits[condition_length], condition_wp_ids, start_indices[condition_length]
                )
            )

        ### Calculate and store metrics for this patient
        _baseline_roc = roc_auc_score(condition_labels, condition_baseline_counts)
        _condition_only_roc = roc_auc_score(condition_labels, condition_only_logits)
        _model_roc = roc_auc_score(condition_labels, condition_subject_logits)

        rocs["baseline"].append(_baseline_roc)
        rocs["condition_only"].append(_condition_only_roc)
        rocs["model"].append(_model_roc)

        _model_precision_at_k = precision_at_k(condition_labels, condition_subject_logits, k)
        _condition_only_precision_at_k = precision_at_k(condition_labels, condition_only_logits, k)
        _baseline_precision_at_k = precision_at_k(condition_labels, condition_baseline_counts, k)

        precisions_at_k["baseline"].append(_baseline_precision_at_k)
        precisions_at_k["condition_only"].append(_condition_only_precision_at_k)
        precisions_at_k["model"].append(_model_precision_at_k)

    ### Computing and print metrics (averaged over patients)

    print("Average model AUC: {}".format(np.average(rocs["model"])))
    print("Average condition only AUC: {}".format(np.average(rocs["condition_only"])))
    print("Average baseline AUC: {}".format(np.average(rocs["baseline"])))

    print("Average model P@K: {}".format(np.average(precisions_at_k["model"])))
    print("Average condition only P@K: {}".format(np.average(precisions_at_k["condition_only"])))
    print("Average baseline P@K: {}".format(np.average(precisions_at_k["baseline"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Location of the model", type=str, required=True)
    parser.add_argument("--tokenizer", help="Location of the tokenizer", type=str, required=True)
    parser.add_argument(
        "--condition-type", choices=["icd9", "stanza"], help="Which condition type to use ?", required=True
    )

    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    model = BertForMaskedLM.from_pretrained(args.model).cuda().eval()
    evaluate(model, tokenizer, args.condition_type)
