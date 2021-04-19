import argparse

import numpy as np
import torch
from experiments.metrics import precision_at_k
from experiments.probing.LR_single_condition_probing import get_frequency_bins, get_non_zero_count_conditions
from experiments.probing.common import generate_name_condition_template
from experiments.utilities import (
    filter_condition_code_by_count,
    get_condition_code_to_count,
    get_condition_code_to_descriptions,
    get_subject_id_to_patient_info,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments

import secrets


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].detach().clone() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(1 if self.labels[idx] else 0)
        return item

    def __len__(self):
        return len(self.labels)


def get_as_dataset(tokenizer, templates, labels):
    batch = tokenizer(
        text=[text.split() for text in templates],
        is_split_into_words=True,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )

    return ClassificationDataset(batch, labels)


def train_model(model, train_dataset, validation_dataset):
    model.train()

    for param in model.base_model.parameters():
        param.requires_grad = True

    dir_ = f"./tmp_{secrets.token_hex(16)}"
    training_args = TrainingArguments(
        output_dir=f"{dir_}/results",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        warmup_steps=500,
        weight_decay=0.001,
        learning_rate=2e-5,
        logging_dir=f"{dir_}/logs",
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        evaluation_strategy="epoch",
        disable_tqdm=True,
    )

    compute_metrics = lambda pred: {"auc": roc_auc_score(pred.label_ids, pred.predictions[:, 1])}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=validation_dataset,
    )

    trainer.train()
    model.eval()

    return trainer


def train_and_evaluate(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    condition_type: str,
    sampling_bin: int,
    n: int,
    metrics_output_path: str,
):
    """Train and evaluate the model on N conditions.
    @param model is the model to encode CLS tokens with.
    @param tokenizer is a BERT tokenizer.
    @param condition_type are we using the icd/medcat extracted conditions?
    @param b is which frequency to sample from.
    @param n is the number of conditions to sample the bin from.
    @return all AUCs and precision @ K scores.
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

    ### Filter condition in each bin so we have atleast one positive training examples
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

        positive_indices = resample(
            positive_indices, replace=True, n_samples=len(negative_indices), random_state=2021
        )
        total_indices = negative_indices + positive_indices

        ### Divide Train Set into Train and Validation Set

        training_indices, validation_indices = train_test_split(
            total_indices, train_size=0.85, random_state=2021, shuffle=True
        )

        # Not too sure we can ensure the validation templates have a positive label in it...
        # Or if there is only 1, that it doesn't end up in the validation set.
        validation_templates = [train_templates[i] for i in validation_indices]
        validation_labels = [train_labels[i] for i in validation_indices]

        np.random.seed(2021)
        np.random.shuffle(training_indices)
        train_templates = [train_templates[i] for i in training_indices]
        train_labels = [train_labels[i] for i in training_indices]

        ### Train the BERT Model

        train_dataset = get_as_dataset(tokenizer, train_templates, train_labels)
        validation_dataset = get_as_dataset(tokenizer, validation_templates, validation_labels)

        clf = train_model(model, train_dataset, validation_dataset)

        ### Get Test Templates

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

        ### Get Test Predictions
        test_dataset = get_as_dataset(tokenizer, test_templates, test_labels)
        test_predictions = clf.predict(test_dataset)

        test_predictions = test_predictions.predictions[:, 1]

        ### Calculate Metrics

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
    parser.add_argument("--model", help="Location of the model", type=str)
    parser.add_argument("--tokenizer", help="Location of the tokenizer", type=str)
    parser.add_argument("--conditions", help="Number of conditions to test per bin", type=int, default=50)
    parser.add_argument("--frequency-bin", help="Which frequency bin to use.", type=int)
    parser.add_argument("--metrics-output-path", help="Where to print results to.", type=str)
    args = parser.parse_args()

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    # Load pre-trained model (weights)
    model = BertForSequenceClassification.from_pretrained(args.model).cuda().eval()

    import os

    metrics_output_path = args.metrics_output_path if args.metrics_output_path is not None else args.model
    metrics_output_path = os.path.join(
        metrics_output_path,
        f"FullBERT_single_conditions_probing/{args.condition_type}_{args.conditions}_{args.frequency_bin}",
    )
    os.makedirs(metrics_output_path, exist_ok=True)

    train_and_evaluate(
        model, tokenizer, args.condition_type, args.frequency_bin, args.conditions, metrics_output_path
    )
