from typing import List

import numpy as np
import torch
from tqdm import tqdm


def get_cls_embeddings(model, tokenizer, templates: List[str], disable_tqdm: bool = False) -> np.ndarray:
    embeddings = []
    batch_size = 2000
    for b in tqdm(range(0, len(templates), batch_size), disable=disable_tqdm):
        batch = templates[b : b + batch_size]
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


def generate_condition_only_template(condition_description: str):
    return f"[CLS] {condition_description.strip()} [SEP]"


def generate_name_condition_template(
    first_name: str, last_name: str, gender: str, condition_description: str
):
    title = "Mr" if gender == "M" else "Mrs"  # I guess just assume married w/e idk ?
    return f"[CLS] {title} {first_name} {last_name} is a yo patient with {condition_description} [SEP]"
