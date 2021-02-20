# TABLE 5
import argparse
from typing import Dict, List

import numpy as np
from experiments.masked_prediction.common import (
    get_condition_predicted_logit,
    get_condition_predicted_rank,
    get_logits_from_templates,
)
from experiments.metrics import precision_at_k
from experiments.utilities import (
    filter_condition_code_by_count,
    get_condition_code_to_count,
    get_condition_code_to_descriptions,
    get_condition_labels_as_vector,
    get_subject_id_to_patient_info,
)
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer


def generate_masked_template(name_length, gender, condition_length):
    title = "Mr" if gender == "M" else "Mrs"  # I guess just assume married w/e idk ?
    name_mask = "[MASK] " * name_length
    condition_mask = "[MASK] " * condition_length
    return f"[CLS] {name_mask.strip()} is a yo patient with {condition_mask.strip()} [SEP]"


def generate_template(name, gender, condition_length):
    title = "Mr" if gender == "M" else "Mrs"  # I guess just assume married w/e idk ?
    condition_mask = "[MASK] " * condition_length
    return f"[CLS] {name} is a yo patient with {condition_mask.strip()} [SEP]"


def evaluate(model, cmp_model, tokenizer, condition_type, metric):
    """Evaluate the performance of the model in terms of being able to predict
    conditons associated with certain names (via templates).
    @param model is the BERT model to encode with.
    @param cmp_model is the BERT model to compare with.
    @param tokenizer is the BERT tokenizer.
    @param stanza is whether or not we use stanza conditions
    @param mode is if we do this normally or mask out everything.
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

    print(len(set_to_use))

    condition_code_to_index: Dict[str, int] = dict(zip(set_to_use, range(len(set_to_use))))

    ### Get list of unique condition lengths (in wordpieces) to generate templates

    condition_code_to_wordpiece_ids: Dict[str, List[str]] = {
        condition: tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(condition_code_to_description[condition])
        )
        for condition in set_to_use
    }  ## Time Saving Measure since we only need condition wordpiece ids

    condition_wordpiece_lengths = [
        len(condition_code_to_wordpiece_ids[condition]) for condition in set_to_use
    ]
    condition_wordpiece_lengths: List[int] = sorted(list(set(condition_wordpiece_lengths)))

    ### Setup metrics

    k = 10
    avg_mroc, avg_mpak = ([], [])
    for subject_id, patient_info in tqdm(subject_id_to_patient_info.items()):
        condition_labels = get_condition_labels_as_vector(patient_info.CONDITIONS, condition_code_to_index)
        condition_labels = np.array(condition_labels)

        if condition_labels.sum() == 0:
            continue  ## Skip if patient is negative for all conditions

        name = patient_info.FIRST_NAME + " " + patient_info.LAST_NAME
        name_length = len(tokenizer.tokenize(name))

        masked_templates = [
            generate_masked_template(name_length, patient_info.GENDER, length)
            for length in condition_wordpiece_lengths
        ]

        named_templates = [
            generate_template(name, patient_info.GENDER, length) for length in condition_wordpiece_lengths
        ]

        start_index = named_templates[0].index("[MASK]")

        named_logits = get_logits_from_templates(
            model,
            tokenizer,
            named_templates,
            normalize=True,
        )

        named_logits = {length: named_logits[i] for i, length in enumerate(condition_wordpiece_lengths)}

        masked_logits = get_logits_from_templates(
            model,
            tokenizer,
            masked_templates,
            normalize=True,
        )

        masked_logits = {length: masked_logits[i] for i, length in enumerate(condition_wordpiece_lengths)}

        score_differences = []

        if metric == "probability":
            metric_calculator = get_condition_predicted_logit
        elif metric == "rank":
            metric_calculator = get_condition_predicted_rank
        else:
            raise NotImplementedError(f"{metric} metric not implemented")

        for condition in set_to_use:
            condition_wp = condition_code_to_wordpiece_ids[condition]
            condition_length = len(condition_wp)
            named_score = metric_calculator(named_logits[condition_length], condition_wp, start_index)
            masked_score = metric_calculator(masked_logits[condition_length], condition_wp, start_index)
            score_differences.append(named_score - masked_score)

        mroc = roc_auc_score(condition_labels, score_differences)
        mpak = precision_at_k(condition_labels, score_differences, k=k)

        avg_mroc.append(mroc)
        avg_mpak.append(mpak)

    print("Average model AUC: {}".format(np.average(avg_mroc)))
    print("Average model P@K: {}".format(np.average(avg_mpak)))    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition-type", choices=["icd9", "stanza"], help="Which condition type to use ?", required=True
    )
    parser.add_argument("--model", help="Location of the model", type=str)
    parser.add_argument("--cmp-model", help="Location of the comparator model.", type=str)
    parser.add_argument("--tokenizer", help="Location of the tokenizer", type=str)
    parser.add_argument(
        "--metric", help="Which metric to calculate delta of ?", type=str, choices=["rank", "probability"]
    )
    args = parser.parse_args()

    # Load pre-trained model tokenizer (vocabulary)
    # '/home/eric/dis_rep/nyu_clincalBERT/clinicalBERT/notebook/bert_uncased/'
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    # Load pre-trained model (weights)
    # '/home/eric/dis_rep/nyu_clincalBERT/convert_to_pytorch/all_useful_100k/'
    model = BertForMaskedLM.from_pretrained(args.model).cuda().eval()
    if args.cmp_model is not None:
        cmp_model = BertForMaskedLM.from_pretrained(args.cmp_model).cuda().eval()
    else:
        cmp_model = BertForMaskedLM.from_pretrained(args.model).cuda().eval()

    evaluate(model, cmp_model, tokenizer, args.condition_type, args.metric)
