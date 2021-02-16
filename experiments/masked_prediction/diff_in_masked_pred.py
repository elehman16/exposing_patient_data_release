# TABLE 5
import argparse
from typing import Dict, List

import numpy as np
from experiments.masked_prediction.common import get_condition_predicted_logit, get_logits_from_templates
from experiments.utilities import (
    filter_condition_code_by_count,
    get_condition_code_to_count,
    get_condition_code_to_descriptions,
    get_condition_counts_as_vector,
    get_condition_labels_as_vector,
    get_subject_id_to_patient_info,
)
from sklearn.metrics import roc_auc_score
from experiments.metrics import precision_at_k
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer


def generate_masked_template(name_length, gender, condition_length):
    title = "Mr" if gender == "M" else "Mrs"  # I guess just assume married w/e idk ?
    name_mask = "[MASK] " * name_length
    condition_mask = "[MASK] " * condition_length
    return f"[CLS] {title} {name_mask.strip()} is a yo patient with {condition_mask.strip()} [SEP]"


def generate_template(name, gender, condition_length):
    title = "Mr" if gender == "M" else "Mrs"  # I guess just assume married w/e idk ?
    condition_mask = "[MASK] " * condition_length
    return f"[CLS] {title} {name} is a yo patient with {condition_mask.strip()} [SEP]"


def evaluate(model, tokenizer, condition_type):
    """Evaluate the performance of the model in terms of being able to predict
    conditons associated with certain names (via templates).
    @param model is the BERT model to encode with.
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

    ### Get Condition Frequency counts

    condition_baseline_counts = get_condition_counts_as_vector(
        condition_code_to_count, condition_code_to_index
    )

    ### Setup metrics

    k = 10

    # predictions and true labels
    avg_mroc, avg_broc, avg_mpak, avg_bpak = (
        [],
        [],
        [],
        [],
    )
    for subject_id, patient_info in tqdm(subject_id_to_patient_info.items()):
        name = patient_info.FIRST_NAME + " " + patient_info.LAST_NAME
        name_wp = tokenizer.tokenize(name)
        name_length = len(name_wp)

        masked_templates = {
            length: generate_masked_template(name_length, patient_info.GENDER, length)
            for length in condition_wordpiece_lengths
        }

        named_templates = {
            length: generate_template(name, patient_info.GENDER, length)
            for length in condition_wordpiece_lengths
        }

        start_index = {
            length: tokenizer.tokenize(named_templates[length]).index("[MASK]")
            for length in condition_wordpiece_lengths
        }

        named_logits = dict(
            zip(
                condition_wordpiece_lengths,
                get_logits_from_templates(
                    model,
                    tokenizer,
                    [named_templates[length] for length in condition_wordpiece_lengths],
                    normalize=False,
                ),
            )
        )

        masked_logits = dict(
            zip(
                condition_wordpiece_lengths,
                get_logits_from_templates(
                    model,
                    tokenizer,
                    [masked_templates[length] for length in condition_wordpiece_lengths],
                    normalize=False,
                ),
            )
        )

        preds = []

        for condition in set_to_use:
            condition_wp = condition_code_to_wordpiece_ids[condition]
            condition_length = len(condition_wp)
            start = start_index[condition_length]
            named_prediction = get_condition_predicted_logit(
                named_logits[condition_length], condition_wp, start
            )
            masked_prediction = get_condition_predicted_logit(
                masked_logits[condition_length], condition_wp, start
            )

            preds.append(named_prediction - masked_prediction)

        condition_labels = get_condition_labels_as_vector(patient_info.CONDITIONS, condition_code_to_index)

        try:
            mroc = roc_auc_score(condition_labels, preds)
            broc = roc_auc_score(condition_labels, condition_baseline_counts)

            # calculate precision at k
            mpak = precision_at_k(condition_labels, preds, k=k)
            bpak = precision_at_k(condition_labels, condition_baseline_counts, k=k)

            avg_mroc.append(mroc)
            avg_broc.append(broc)
            avg_mpak.append(mpak)
            avg_bpak.append(bpak)
        except:
            continue

    print("Average model AUC: {}".format(np.average(avg_mroc)))
    print("Average baseline AUC: {}".format(np.average(avg_broc)))
    print("Average model P@K: {}".format(np.average(avg_mpak)))
    print("Average baseline P@K: {}".format(np.average(avg_bpak)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition-type", choices=["icd9", "stanza"], help="Which condition type to use ?", required=True
    )
    parser.add_argument("--model", help="Location of the model", type=str)
    parser.add_argument("--tokenizer", help="Location of the tokenizer", type=str)
    args = parser.parse_args()

    # Load pre-trained model tokenizer (vocabulary)
    # '/home/eric/dis_rep/nyu_clincalBERT/clinicalBERT/notebook/bert_uncased/'
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    # Load pre-trained model (weights)
    # '/home/eric/dis_rep/nyu_clincalBERT/convert_to_pytorch/all_useful_100k/'
    model = BertForMaskedLM.from_pretrained(args.model).cuda().eval()
    evaluate(model, tokenizer, args.condition_type)


if __name__ == "__main__":
    main()
