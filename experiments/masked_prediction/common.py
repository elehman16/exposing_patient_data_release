from typing import Callable, Dict, List

import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer


def get_logits_from_templates(
    model: BertForMaskedLM, tokenizer: BertTokenizer, templates: List[str], normalize: bool = False
) -> List[np.ndarray]:
    torch.cuda.empty_cache()
    split_texts = [template.split() for template in templates]
    batch = tokenizer(
        text=split_texts,
        is_split_into_words=True,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )

    with torch.no_grad():
        predictions = model(batch.input_ids.cuda(), attention_mask=batch.attention_mask.cuda())
        if normalize :
            logits = torch.nn.Softmax(-1)(predictions.logits)
        else :
            logits = predictions.logits
            
        logits = logits.cpu().data.numpy()

        mask = batch.attention_mask.sum(-1)
        logits = [logits[i, : mask[i]] for i in range(mask.shape[0])]

    return logits


def get_average_predicted_score(
    logits: np.ndarray, wordpiece_ids: List[int], start_index: int
) -> float:
    """
    Predict the mask tokens in the template given/our standard template.
    @param condition the condition that we want to inquire if this person has.
    @param logits: dict mapping condition length to template querying that condition length
    @return the prediction logit for the condition (average logits of wordpieces in the condition).
    """

    prediction_value = 0
    for i, token_index in enumerate(wordpiece_ids):
        wordpiece_logit = logits[start_index + i][token_index]
        prediction_value += wordpiece_logit 

    return float(prediction_value) / len(wordpiece_ids)

def get_average_predicted_rank(
    logits: np.ndarray, wordpiece_ids: List[int], start_index: int
) -> float:
    """
    Predict the mask tokens in the template given/our standard template.
    @param condition the condition that we want to inquire if this person has.
    @param logits: dict mapping condition length to template querying that condition length
    @return the prediction logit for the condition (average logits of wordpieces in the condition).
    """

    prediction_value = 0
    for i, token_index in enumerate(wordpiece_ids):
        wordpiece_logit = logits[start_index + i][token_index]
        wordpiece_rank = (logits[start_index +i] > wordpiece_logit).sum()
        prediction_value += wordpiece_rank / len(wordpiece_ids)

    return float(prediction_value)

def get_scoring_function(metric: str) -> Callable[[np.ndarray, List[int], int], float]:
    if metric in ["logit", "probability"] :
        return get_average_predicted_score
    elif metric == "rank" :
        return get_average_predicted_rank
    else :
        raise NotImplementedError(f"{metric} function is not implemented")