import argparse

import numpy as np
from experiments.metrics import precision_at_k
from experiments.probing.probing_utils import get_cls_embeddings
from experiments.utilities import get_patient_name_to_is_modified
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer


def generate_name_templates(name):
    return f"[CLS] {name} [SEP]"


def train_and_evaluate(model, tokenizer):
    """Train and evaluate the model.
    @param model is the BERT model.
    @param tokenizer is the BERT tokenizer.
    @param model_save_location is the save location of the model
    """
    patient_name_to_modified = get_patient_name_to_is_modified()

    print(np.array(list(patient_name_to_modified.values())).mean())

    names = list(patient_name_to_modified.keys())

    train_names, test_names = train_test_split(names, train_size=0.5, random_state=2021, shuffle=True)

    train_templates = [generate_name_templates(name) for name in train_names]
    train_labels = [patient_name_to_modified[name] for name in train_names]

    train_embeddings = get_cls_embeddings(model, tokenizer, train_templates)

    clf = LogisticRegression(random_state=2021, max_iter=10000).fit(train_embeddings, train_labels)

    test_templates = [generate_name_templates(name) for name in test_names]
    test_labels = [patient_name_to_modified[name] for name in test_names]

    test_embeddings = get_cls_embeddings(model, tokenizer, test_templates)

    test_predictions = clf.predict_proba(test_embeddings)[:, 1]

    auc_score = roc_auc_score(test_labels, test_predictions)
    precision_at_10 = precision_at_k(test_labels, test_predictions, k=10)
    precision_at_50 = precision_at_k(test_labels, test_predictions, k=50)

    print(f"AUC Score : {auc_score}")
    print(f"P@10 {precision_at_10}")
    print(f"P@50 {precision_at_50}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Location of the model", type=str)
    parser.add_argument("--tokenizer", help="Location of the tokenizer", type=str)
    args = parser.parse_args()

    # Load pre-trained model tokenizer (vocabulary)
    # '/home/eric/dis_rep/nyu_clincalBERT/clinicalBERT/notebook/bert_uncased/'
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    # Load pre-trained model (weights)
    # '/home/eric/dis_rep/nyu_clincalBERT/convert_to_pytorch/all_useful_100k/'
    model = BertModel.from_pretrained(args.model).cuda().eval()
    train_and_evaluate(model, tokenizer)


if __name__ == "__main__":
    main()