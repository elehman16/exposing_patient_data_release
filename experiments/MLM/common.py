from typing import Callable, List

import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer


def get_logits_from_templates(
    model: BertForMaskedLM,
    tokenizer: BertTokenizer,
    templates: List[str],
    normalize: bool = False,
    temperature: float = 1.0,
) -> List[np.ndarray]:
    """Return a list of numpy arrays of same size as templates list,
    corresponding to token MLM scores for each template.
    ndarray at position i in the list is of shape (#Wordpiece in templates[i], #Vocab_Size ~ 30K)

    ### Args:
        normalize: Token scores are either logits (normalize=False) or probabilities (normalize=True)
    """

    # assert normalize, "Logits not normalized "
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
        mask = batch.attention_mask.sum(-1)

        if normalize:
            logits = torch.nn.LogSoftmax(dim=-1)(predictions.logits / temperature)
        else:
            logits = predictions.logits

        logits = logits.cpu().data.numpy()
        logits = [logits[i, : mask[i]] for i in range(mask.shape[0])]

    return logits


def get_average_predicted_score(
    logits: np.ndarray, target_wordpiece_ids: List[int], start_index: int
) -> float:
    """Given a `logits` array of shape (Number of wordpieces, vocab size), return average
    score for the `target_wordpiece_ids`, with positions in logits array starting at `start_index`.
    """

    prediction_value = 0
    for i, token_index in enumerate(target_wordpiece_ids):
        wordpiece_logit = logits[start_index + i][token_index]
        prediction_value += wordpiece_logit

    return float(prediction_value) / len(target_wordpiece_ids)


def get_average_predicted_rank(
    logits: np.ndarray, target_wordpiece_ids: List[int], start_index: int
) -> float:
    """Given a `logits` array of shape (Number of wordpieces, vocab size), return average
    rank for the `target_wordpiece_ids`, with positions in logits array starting at `start_index`.
    """

    prediction_value = 0
    for i, token_index in enumerate(target_wordpiece_ids):
        wordpiece_logit = logits[start_index + i][token_index]
        wordpiece_rank = (logits[start_index + i] > wordpiece_logit).sum()
        prediction_value += wordpiece_rank / len(target_wordpiece_ids)

    return float(prediction_value)


def get_scoring_function(metric: str) -> Callable[[np.ndarray, List[int], int], float]:
    """Returns either function `get_average_predicted_rank` or `get_average_predicted_score` on basis of
    metric (takes value in [rank, probability]).

    Either function has same declaration
    ```python
    function_name(logits: np.ndarray, target_wordpiece_ids: List[int], start_index: int)
    ```

    -- Given a `logits` array of shape (Number of wordpieces, vocab size), return average
    rank/score for the `target_wordpiece_ids`, with positions in logits array starting at `start_index`.
    """
    if metric in ["logit", "probability"]:
        return get_average_predicted_score
    elif metric == "rank":
        return get_average_predicted_rank
    else:
        raise NotImplementedError(f"{metric} function is not implemented")


def mean_std_as_string(metric_name, values):
    return f"{metric_name}: Mean {np.average(values)} Std {np.std(values)}\n"
