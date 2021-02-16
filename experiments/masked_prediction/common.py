from typing import Dict, List

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


def get_condition_predicted_logit(
    logits: np.ndarray, condition_wordpiece_ids: List[int], start_index: int
) -> float:
    """
    Predict the mask tokens in the template given/our standard template.
    @param condition the condition that we want to inquire if this person has.
    @param logits: dict mapping condition length to template querying that condition length
    @return the prediction logit for the condition (average logits of wordpieces in the condition).
    """

    prediction_value = 0
    for i, token_index in enumerate(condition_wordpiece_ids):
        wordpiece_logit = logits[start_index + i][token_index]
        prediction_value += wordpiece_logit / len(condition_wordpiece_ids)

    return float(prediction_value)
