import glob

import config
import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizerFast
import torch

from joblib import Parallel, delayed

from typing import List

nlp = spacy.load("en")

import subprocess

from argparse import ArgumentParser

from sklearn.metrics import roc_auc_score
from experiments.metrics import precision_at_k

import sys


def batched_perplexity(
    model: BertForMaskedLM,
    tokenizer: BertTokenizerFast,
    split_sentences: List[List[str]],
    mask: List[List[bool]],
) -> np.ndarray:
    ## Since spacy work at word level while bert at wordpiece level, change is needed

    batch_size = 256

    losses = []

    for batch_idx in tqdm(range(0, len(split_sentences), batch_size)):
        loss = perplexity(model, tokenizer, split_sentences, mask, batch_size, batch_idx)
        losses.append(loss)

    return np.concatenate(losses)


def perplexity(model, tokenizer, split_sentences, mask, batch_size, batch_idx):
    torch.cuda.empty_cache()
    batch_encoding = tokenizer(
        split_sentences[batch_idx : batch_idx + batch_size],
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    batch_mask = np.zeros_like(batch_encoding.input_ids.cpu().data.numpy())

    for i, token_mask in enumerate(mask[batch_idx : batch_idx + batch_size]):
        masked_token_indices = [i for i, v in enumerate(token_mask) if v]
        for index in masked_token_indices:
            wordpiece_span = batch_encoding.word_to_tokens(i, index)
            batch_mask[i, wordpiece_span.start : wordpiece_span.end] = 1

    batch_mask = torch.tensor(batch_mask == 1).to(model.device)
    batch_labels = batch_encoding.input_ids.detach().clone().to(model.device)
    batch_labels.masked_fill_(~batch_mask, -100)
    batch_encoding.input_ids.masked_fill_(batch_mask, 103)

    with torch.no_grad():
        output = model(**batch_encoding)
        loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
            output.logits.transpose(1, 2).contiguous(), batch_labels
        )
        assert loss.shape == batch_labels.shape
        loss = loss.sum(-1) / batch_mask.sum(-1)
        loss = loss.detach().cpu().data.numpy()

    return loss


def ner_tag_names(text):
    text = text.replace("[CLS]", "").replace("[SEP]", "").strip()
    doc = nlp(text)

    tokens = [token.text for token in doc]
    is_name = [token.ent_type_ == "PERSON" for token in doc]

    name_group_index = [i for i, v in enumerate(is_name) if v]
    created_samples = []

    for name_index in name_group_index:
        created_samples.append((tokens, tokens[name_index], [i == name_index for i in range(len(tokens))]))

    return created_samples


def batched_ner_tag_names(texts):
    return [ner_tag_names(text) for text in texts]


def preprocess_parallel(texts, chunksize=100):
    chunker = (texts[i : i + chunksize] for i in range(0, len(texts), chunksize))
    executor = Parallel(n_jobs=28, backend="multiprocessing", prefer="processes", verbose=20)
    do = delayed(batched_ner_tag_names)
    tasks = (do(chunk) for chunk in chunker)
    result = executor(tasks)
    return [text for chunk in result for text in chunk]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", help="Location of the model.", type=str, required=True)
    parser.add_argument("--tokenizer", help="Location of the tokenizer.", type=str, required=True)
    parser.add_argument("--comparator", help="Location of the comparator model.", type=str, required=True)
    parser.add_argument("--sample-files", help="Location of the file(s) containing generated text.", type=str, required=True)
    parser.add_argument("--metrics-output-path", help="Where to print the results.", type=str, required=True)
    args = parser.parse_args()

    txt_files = glob.glob(args.sample_files)

    all_sentences = []

    for f in tqdm(txt_files):
        with open(f) as tmp:
            all_sentences += [line.strip() for line in tmp]

    samples = preprocess_parallel(all_sentences, 1000)

    num_sentences_with_samples = [len(s) > 0 for s in samples]
    lengths = [len(s) for s in samples if len(s) > 0]
    print(f"{np.mean(num_sentences_with_samples)} -- {np.mean(lengths)}")

    all_samples = [sample for sentence_samples in samples for sample in sentence_samples]
    sample_sentences, sample_names, sample_masks = list(zip(*all_samples))
    sample_sentences, sample_names, sample_masks = (
        list(sample_sentences),
        list(sample_names),
        list(sample_masks),
    )

    print(len(sample_sentences))

    sys.exit(0)

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)
    model = BertForMaskedLM.from_pretrained(args.model).eval().cuda()
    cmp_model = BertForMaskedLM.from_pretrained(args.comparator).eval().cuda()

    metrics_output_path = args.model if args.metrics_output_path is None else args.metrics_output_path
    print(f"Saving results to {metrics_output_path}")
    

    torch.cuda.empty_cache()
    losses_under_model = batched_perplexity(model, tokenizer, sample_sentences, sample_masks)
    torch.cuda.empty_cache()
    losses_under_comparator = batched_perplexity(cmp_model, tokenizer, sample_sentences, sample_masks)

    loss_diff = losses_under_comparator - losses_under_model

    df = pd.read_csv(config.SUBJECT_ID_to_NAME)
    modified = set(pd.read_csv(config.MODIFIED_SUBJECT_IDS)["SUBJECT_ID"])
    df = df[df["SUBJECT_ID"].isin(modified)]

    # Lower case the first and last names
    df["FIRST_NAME"] = df["FIRST_NAME"].apply(lambda x: str(x).lower())
    df["LAST_NAME"] = df["LAST_NAME"].apply(lambda x: str(x).lower())

    first_names = set(df["FIRST_NAME"].values)
    last_names = set(df["LAST_NAME"].values)

    sample_names_set = sorted(list(set(sample_names)))

    is_known_first_name = np.array([name.lower() in first_names for name in sample_names_set])
    is_known_last_name = np.array([name.lower() in last_names for name in sample_names_set])

    is_known_name = np.logical_or(is_known_first_name, is_known_last_name)

    loss_diff_set = {name: 0 for name in sample_names_set}
    for i, name in enumerate(sample_names):
        loss_diff_set[name] = max(loss_diff_set[name], loss_diff[i])

    loss_diff = np.array([loss_diff_set[name] for name in sample_names_set])

    with open(f"{metrics_output_path}/sampling_results.txt", "w") as f:
        f.write(f"#generated : {len(sample_names_set)} -- {len(sample_names)}\n")
        f.write(f"#First Names : {np.average(is_known_first_name)}\n")
        f.write(f"#Last Names : {np.average(is_known_last_name)}\n")

        f.write(f"AUC : {roc_auc_score(is_known_name, loss_diff)}\n")
        f.write(f"P@100 : {precision_at_k(is_known_name, loss_diff, k=100)}\n")
