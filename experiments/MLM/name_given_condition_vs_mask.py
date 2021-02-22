# TABLE 5
import argparse
from typing import Dict, List

import numpy as np
from experiments.MLM.common import (
    get_scoring_function, get_logits_from_templates)
from experiments.metrics import precision_at_k
from experiments.utilities import (filter_condition_code_by_count,
                                   get_condition_code_to_count,
                                   get_condition_code_to_descriptions,
                                   get_condition_labels_as_vector,
                                   get_subject_id_to_patient_info)
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer


def generate_masked_template(name_length, condition_length):
    # title = "Mr" if gender == "M" else "Mrs"  # I guess just assume married w/e idk ?
    name_mask = "[MASK] " * name_length
    condition_mask = "[MASK] " * condition_length
    return f"[CLS] {name_mask.strip()} is a yo patient with {condition_mask.strip()} [SEP]"


def generate_template(name_length, condition):
    # title = "Mr" if gender == "M" else "Mrs"  # I guess just assume married w/e idk ?
    name_mask = "[MASK] " * name_length
    return f"[CLS] {name_mask.strip()} is a yo patient with {condition} [SEP]"


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
        set_to_use = filter_condition_code_by_count(condition_code_to_count, min_count=0, max_count=float("inf"))
    else:
        raise NotImplementedError()

    print(len(set_to_use))

    condition_code_to_index: Dict[str, int] = dict(zip(set_to_use, range(len(set_to_use))))

    ### Get list of unique name lengths (in wordpieces) to generate templates

    subject_id_to_name_wordpiece_ids: Dict[str, List[int]] = {
        subject_id: tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(patient_info.FIRST_NAME + " " + patient_info.LAST_NAME)
        )
        for subject_id, patient_info in subject_id_to_patient_info.items()
    }  ## Time Saving Measure since we only need condition wordpiece ids

    name_wordpiece_lengths = [len(wordpieces) for wordpieces in subject_id_to_name_wordpiece_ids.values()]
    name_wordpiece_lengths: List[int] = sorted(list(set(name_wordpiece_lengths)))

    ### Setup metrics

    avg_mroc, avg_mpak = [], []

    preds = {subject_id: [] for subject_id in subject_id_to_name_wordpiece_ids.keys()}

    for condition in tqdm(set_to_use):
        condition_desc = condition_code_to_description[condition]
        condition_length = len(tokenizer.tokenize(condition_desc))

        masked_templates = {
            length: generate_masked_template(length, condition_length) for length in name_wordpiece_lengths
        }

        named_templates = {
            length: generate_template(length, condition_desc) for length in name_wordpiece_lengths
        }

        start_index = {
            length: tokenizer.tokenize(named_templates[length]).index("[MASK]")
            for length in name_wordpiece_lengths
        }

        named_logits = get_logits_from_templates(
            model,
            tokenizer,
            [named_templates[length] for length in name_wordpiece_lengths],
            normalize=True,
        )

        named_logits = {length: named_logits[i] for i, length in enumerate(name_wordpiece_lengths)}

        masked_logits = get_logits_from_templates(
            model,
            tokenizer,
            [masked_templates[length] for length in name_wordpiece_lengths],
            normalize=True,
        )

        masked_logits = {length: masked_logits[i] for i, length in enumerate(name_wordpiece_lengths)}

        scoring_function = get_scoring_function(metric)

        for subject_id, name_wp in subject_id_to_name_wordpiece_ids.items():
            name_length = len(name_wp)
            start = start_index[name_length]

            named_score = scoring_function(named_logits[name_length], name_wp, start)
            masked_score = scoring_function(masked_logits[name_length], name_wp, start)

            preds[subject_id].append(named_score - masked_score)

    for subject_id, patient_info in tqdm(subject_id_to_patient_info.items()):
        condition_labels = get_condition_labels_as_vector(patient_info.CONDITIONS, condition_code_to_index)
        subject_preds = preds[subject_id]
        try:
            mroc = roc_auc_score(condition_labels, subject_preds)
            mpak = precision_at_k(condition_labels, subject_preds, k=10)

            avg_mroc.append(mroc)
            avg_mpak.append(mpak)
        except:
            continue

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
    parser.add_argument("--metric", help="Which metric to calculate delta of ?", type=str, choices=["rank", "probability"])
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
