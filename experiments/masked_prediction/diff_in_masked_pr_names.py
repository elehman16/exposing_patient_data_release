import torch
import argparse
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.metrics import roc_auc_score, ndcg_score, precision_score
import numpy as np
from tqdm import tqdm

from experiments.utilities import get_patient_name_to_is_modified

### Here, we are trying to run experiments for figure 4, which
# is when we have either the first or last name, and try to predict
# the other.
sm = torch.nn.Softmax(dim=-1)

def get_probability(model, tokenizer, tokenized_text, start, end, tokenized_string):
    # Create the segments tensors.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens]).cuda()
    segments_tensors = torch.tensor([segments_ids]).cuda()

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)

    idx, prediction_value = 0, 0
    indices_condition = tokenizer.convert_tokens_to_ids(tokenized_string)
    for i, masked_index in enumerate(range(start, end)):
        idx_correct_label = indices_condition[idx]
        wp = sm(predictions[0][masked_index])[idx_correct_label]
        prediction_value += wp.detach().item() / len(tokenized_text)
        idx += 1
    
    return prediction_value

def get_rank(model, tokenizer, tokenized_text, start, end, tokenized_string):
    # Create the segments tensors
    segments_ids = [0] * len(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens]).cuda()
    segments_tensors = torch.tensor([segments_ids]).cuda()

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)

    idx, avg_rank = 0, 0
    indices_condition = tokenizer.convert_tokens_to_ids(tokenized_string)
    preds = predictions[0][start:end]
    for i in range(end - start):
        pr = preds[i][indices_condition[i]]
        avg_rank += sum(preds[i] > pr) / (end - start)

    return avg_rank


def get_rank_diff(model, tokenizer, template, masked_template, masked_string):
    tokenized_string = tokenizer.tokenize(masked_string)
    tokenized_temp = tokenizer.tokenize(template)
    tokenized_mask_temp = tokenizer.tokenize(masked_template)
    st_, end = tokenized_temp.index("[MASK]"), tokenized_temp.index("[MASK]") + len(tokenized_string)
    r1 = get_rank(model, tokenizer, tokenized_temp, st_, end, tokenized_string)
    r2 = get_rank(model, tokenizer, tokenized_mask_temp, st_, end, tokenized_string)

    return r1 - r2


def generate_masked_template(tokenizer, to_mask):
    """Generate a template given the information given.
    @param tokenizer is the tokenizer for the model.
    @param to_mask is the item to mask
    @return the template to be encoded (with MASKs).
    """
    return "[CLS] {} [SEP]".format("[MASK] " * len(tokenizer.tokenize(to_mask)))


def generate_template(tokenizer, first_name, last_name, mode):
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


def evaluate(model, tokenizer, mode):
    """Evaluate the performance of the model in terms of being able to predict
    conditons associated with certain names (via templates).
    @param model is the BERT model to encode with.
    @param tokenizer is the BERT tokenizer.
    @param stanza is whether or not we use stanza conditions
    @param mode is if we do this normally or mask out everything.
    """
    avg_diff = []
    patient_name_to_modified = get_patient_name_to_is_modified()

    for name, is_modified in tqdm(patient_name_to_modified.items()):
        first_name, last_name = name.split(" ", 1)
        to_predict = first_name if mode == "mask_first" else last_name
        template = generate_template(tokenizer, first_name, last_name, mode)
        masked_template = generate_masked_template(tokenizer, name)

        avg_diff.append(get_rank_diff(model, tokenizer, template, masked_template, to_predict))

    print(np.average(avg_diff))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Location of the model", type=str)
    parser.add_argument("--tokenizer", help="Location of the tokenizer", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--metric", type=str, choices=["rank", "probability"])
    args = parser.parse_args()

    # Load pre-trained model tokenizer (vocabulary)
    # '/home/eric/dis_rep/nyu_clincalBERT/clinicalBERT/notebook/bert_uncased/'
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    # Load pre-trained model (weights)
    # '/home/eric/dis_rep/nyu_clincalBERT/convert_to_pytorch/all_useful_100k/'
    model = BertForMaskedLM.from_pretrained(args.model).cuda().eval()
    evaluate(model, tokenizer, args.mode)

if __name__ == "__main__":
    main()
