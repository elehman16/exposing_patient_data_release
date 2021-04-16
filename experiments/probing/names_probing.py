import argparse

import numpy as np
from experiments.metrics import precision_at_k
from experiments.probing.common import get_cls_embeddings
from experiments.utilities import get_patient_name_to_is_reidentified
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer


def generate_name_templates(name):
    return f"[CLS] {name} [SEP]"


def train_and_evaluate(model, tokenizer, metrics_output_path):
    """Train and evaluate the model.

    Train a LR to distinguish between names appearing in the text BERT was trained on
    from those that didn't appear.
    """
    patient_name_to_reidentified = get_patient_name_to_is_reidentified()

    print(np.array(list(patient_name_to_reidentified.values())).mean())

    names = list(patient_name_to_reidentified.keys())

    train_names, test_names = train_test_split(names, train_size=0.5, random_state=2021, shuffle=True)

    train_templates = [generate_name_templates(name) for name in train_names]
    train_labels = [patient_name_to_reidentified[name] for name in train_names]

    train_embeddings = get_cls_embeddings(model, tokenizer, train_templates)

    clf = LogisticRegression(random_state=2021, max_iter=10000).fit(train_embeddings, train_labels)

    test_templates = [generate_name_templates(name) for name in test_names]
    test_labels = [patient_name_to_reidentified[name] for name in test_names]

    test_embeddings = get_cls_embeddings(model, tokenizer, test_templates)

    test_predictions = clf.predict_proba(test_embeddings)[:, 1]

    auc_score = roc_auc_score(test_labels, test_predictions)
    precision_at_10 = precision_at_k(test_labels, test_predictions, k=10)
    precision_at_50 = precision_at_k(test_labels, test_predictions, k=50)

    with open(f"{metrics_output_path}/results.txt", "w") as f:
        f.write(f"Model AUC {auc_score}\n")
        f.write(f"Model P@10 {precision_at_10}\n")
        f.write(f"Model P@50 {precision_at_50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Location of the model", type=str, required=True)
    parser.add_argument("--tokenizer", help="Location of the tokenizer", type=str, required=True)
    parser.add_argument("--metrics-output-path", type=str)
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    model = BertModel.from_pretrained(args.model).cuda().eval()

    import os
    metrics_output_path = args.metrics_output_path if args.metrics_output_path is not None else args.model
    metrics_output_path = os.path.join(metrics_output_path, f"names_probing/")
    os.makedirs(metrics_output_path, exist_ok=True)

    train_and_evaluate(model, tokenizer, metrics_output_path)
