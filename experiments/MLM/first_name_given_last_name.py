import argparse
from typing import List

import numpy as np
from experiments.MLM.common import (get_logits_from_templates,
                                    get_scoring_function)
from experiments.utilities import get_patient_name_to_is_reidentified
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer


def generate_masked_template(tokenizer: BertTokenizer, name: str) -> str:
    return "[CLS] {} [SEP]".format("[MASK] " * len(tokenizer.tokenize(name)))


def generate_template(tokenizer: BertTokenizer, first_name: str, last_name: str, mode: str) -> str:
    """Generate a template given the information given.
    @param tokenizer is the tokenizer for the model.
    @param first_name is the patient's first name
    @param last_name is the patient's last name.
    @param mode will determine if we mask out first or last name.
    @return the template to be encoded (with MASKs).
    """
    if mode == "mask_first":
        tok_name = tokenizer.tokenize(first_name)
        mask_string = "[MASK] " * len(tok_name)
        return f"[CLS] {mask_string.strip()} {last_name} [SEP]"
    elif mode == "mask_last":
        tok_name = tokenizer.tokenize(last_name)
        mask_string = "[MASK] " * len(tok_name)
        return f"[CLS] {first_name} {mask_string.strip()} [SEP]"


def evaluate(model: BertForMaskedLM, tokenizer: BertTokenizer, mode: str, metric: str, metrics_output_path):
    """Evaluate the performance of the model in terms of being able to predict
    conditons associated with certain names (via templates).
    @param model is the BERT model to encode with.
    @param cmp_model is the comparator BERT model (same tokenizer).
    @param tokenizer is the BERT tokenizer.
    @param stanza is whether or not we use stanza conditions
    @param mode is if we do this normally or mask out everything.
    """

    patient_name_to_reidentified = get_patient_name_to_is_reidentified()

    filled_templates = []
    masked_templates = []
    target_ids_list = []
    start_indices = []
    labels = []

    for name, is_reidentified in tqdm(patient_name_to_reidentified.items()):
        first_name, last_name = name.split(" ", 1)
        if len(first_name) == 0 or len(last_name) == 0:
            continue

        filled_template = generate_template(tokenizer, first_name, last_name, mode)
        filled_templates.append(filled_template)
        masked_templates.append(generate_masked_template(tokenizer, name))

        assert len(tokenizer.tokenize(filled_templates[-1])) == len(
            tokenizer.tokenize(masked_templates[-1])
        ), breakpoint()

        target = first_name if mode == "mask_first" else last_name
        target_ids: List[int] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target))
        target_ids_list.append(target_ids)

        start_index = tokenizer.tokenize(filled_template).index("[MASK]")
        start_indices.append(start_index)

        labels.append(is_reidentified)

    scoring_function = get_scoring_function(metric)

    score_difference = []

    batch_size = 100
    for b in tqdm(range(0, len(labels), batch_size)):
        filled_logits = get_logits_from_templates(
            model, tokenizer, filled_templates[b : b + batch_size], normalize=True
        )
        masked_logits = get_logits_from_templates(
            model, tokenizer, masked_templates[b : b + batch_size], normalize=True
        )

        batch_start_indices = start_indices[b : b + batch_size]
        batch_target_ids_list = target_ids_list[b : b + batch_size]

        for i in range(len(batch_start_indices)):
            filled_score = scoring_function(
                filled_logits[i], batch_target_ids_list[i], batch_start_indices[i]
            )
            masked_score = scoring_function(
                masked_logits[i], batch_target_ids_list[i], batch_start_indices[i]
            )
            score_difference.append(np.exp(filled_score) - np.exp(masked_score))

    score_difference = np.array(score_difference)
    labels = np.array(labels)

    from sklearn.metrics import roc_auc_score
    from experiments.metrics import precision_at_k

    with open(f"{metrics_output_path}/results.txt", "w") as f :
        f.write(roc_auc_score(labels, score_difference))  
        f.write(precision_at_k(labels, score_difference, k=50))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Location of the model", type=str, required=True)
    parser.add_argument("--tokenizer", help="Location of the tokenizer", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["mask_first", "mask_last"], required=True)
    parser.add_argument(
        "--metric", help="Which metric to calculate ?", choices=["rank", "probability"], required=True
    )
    parser.add_argument("--metrics-output-path", type=str)
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    model = BertForMaskedLM.from_pretrained(args.model).cuda().eval()

    metrics_output_path = args.metrics_output_path if args.metrics_output_path is not None else args.model

    import os
    metrics_output_path = os.path.join(metrics_output_path, f"first_name_given_last_name/{args.mode}_{args.metric}")
    os.makedirs(metrics_output_path, exist_ok=True)

    evaluate(model, tokenizer, args.mode, args.metric, metrics_output_path)
