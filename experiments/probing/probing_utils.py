import numpy as np
import torch 
def per_syllabal(tokenizer, set_to_use):
    """ Split Conditions into word pieces.
    @param tokeinzer is the BERT tokenizer.
    @param set_to_use is the set of conditions to consider (i.e. our universe of conditions).
    @return a dictionary that maps number of a word piece tokens for a condition
    to a list of conditions.
    """
    lens_ = {}
    for c in set_to_use:
        tok_c = tokenizer.tokenize(c)
        if len(tok_c) in lens_:
            lens_[len(tok_c)].append(c)
        else:
            lens_[len(tok_c)] = [c]

    return lens_

def balance_classes(x, y, subsample_size=1):
    """ """
    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    print("Number of 1 samples: {} / {}".format(list(ys).count(1), len(ys)))
    return xs,ys

def get_cls_condition_representation(model, tokenizer, gen_temp):
    """ Given a name and a condition, what is the prediction value of the condition?
    @param model is the model to encode the template with.
    @param tokenizer is the BERT tokenizer.
    @param gen_temp is the template to encode.
    @return a CLS encoded representation of the data
    """
    segments_ids = [0] * len(gen_temp[0])
    indexed_tokens = [tokenizer.convert_tokens_to_ids(gt) for gt in gen_temp]

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor(indexed_tokens).cuda()
    segments_tensors = torch.tensor([segments_ids] * len(indexed_tokens)).cuda()

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)

    prediction_values = predictions[0][-2][:,0]
    return prediction_values.cpu().numpy()

from typing import List 
from tqdm import tqdm

def get_cls_embeddings(model, tokenizer, templates: List[str], disable_tqdm: bool = False) -> np.ndarray:
    embeddings = []
    batch_size = 500
    for b in tqdm(range(0, len(templates), batch_size), disable=disable_tqdm) :
        batch = templates[b:b + batch_size]
        # torch.cuda.empty_cache()
        split_texts = [template.split() for template in batch]
        batch = tokenizer(
            text=split_texts,
            is_split_into_words=True,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        with torch.no_grad():
            predictions = model(batch.input_ids.cuda(), attention_mask=batch.attention_mask.cuda())
            cls_embeddings = predictions.pooler_output.cpu().data.numpy()
            embeddings.append(cls_embeddings)

    return np.concatenate(embeddings, axis=0)