import argparse
from typing import Dict, List

import numpy as np
import torch
from experiments.metrics import differential_similarity
from experiments.utilities import (
    filter_condition_code_by_count,
    get_condition_code_to_count,
    get_condition_code_to_descriptions,
    get_condition_labels_as_vector,
    get_subject_id_to_patient_info,
)
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast

normalize = lambda x: torch.nn.functional.normalize(x, dim=-1)
cosine_sim = lambda x, y: (normalize(x) * normalize(y)).sum(-1)


def generate_template(first_name, last_name, gender, condition_description):
    title = "Mr" if gender == "M" else "Mrs"  # I guess just assume married w/e idk ?
    return f"[CLS] {title} {first_name} {last_name} is a yo patient with {condition_description} [SEP]"


def get_name_condition_similarities(
    model: BertModel,
    tokenizer: BertTokenizerFast,
    templates: List[str],
    name_start_index: int,
    name_end_index: int,
    condition_start_index: int,
    condition_end_index: int,
):
    mean_similarities = []
    max_similarities = []
    all_pair_similarities = []

    batch_size = 3000
    for b in tqdm(range(0, len(templates), batch_size), disable=True):
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
            predictions = model(
                batch.input_ids.cuda(), attention_mask=batch.attention_mask.cuda(), output_hidden_states=True
            )
            hidden_states = predictions.last_hidden_state  # (B, L, H)
            template_lengths = batch.attention_mask.sum(-1)

            name_embeddings = hidden_states[:, name_start_index:name_end_index]  # (B, L_name, H)

            # (B, L_cond, H) This is necessary since each condition has difference wordpiece length.
            # Therefore, we can keep it as tensor anymore.
            condition_embeddings = []  

            for i in range(hidden_states.shape[0]):
                template_hidden_states = hidden_states[i, : template_lengths[i]]
                condition_embeddings.append(template_hidden_states[condition_start_index:condition_end_index])

            mean_condition_embeddings = torch.cat(
                [embedding.mean(0)[None, :] for embedding in condition_embeddings], dim=0
            )  # (B, H)
            similarity = cosine_sim(name_embeddings.mean(1), mean_condition_embeddings)  # (B, )
            mean_similarities.append(similarity.cpu().data.numpy())

            max_condition_embeddings = torch.cat(
                [embedding.max(0).values[None, :] for embedding in condition_embeddings], dim=0
            )  # (B, H)
            similarity = cosine_sim(name_embeddings.max(1).values, max_condition_embeddings)
            max_similarities.append(similarity.cpu().data.numpy())

            for i, embedding in enumerate(condition_embeddings):
                similarity_matrix = normalize(embedding) @ normalize(name_embeddings[i]).T
                all_pair_similarities.append(torch.max(similarity_matrix).cpu().data.numpy())

    return (
        np.concatenate(mean_similarities, axis=0),
        np.concatenate(max_similarities, axis=0),
        np.array(all_pair_similarities),
    )


def main(model, tokenizer: BertTokenizerFast, condition_type: str):
    """Compute the BERT representations + cosine similarities."""

    ## Get Relevant data

    subject_id_to_patient_info = get_subject_id_to_patient_info(condition_type=condition_type)
    condition_code_to_count = get_condition_code_to_count(condition_type=condition_type)
    condition_code_to_description = get_condition_code_to_descriptions(condition_type=condition_type)

    if condition_type == "stanza":
        set_to_use = filter_condition_code_by_count(condition_code_to_count, min_count=50, max_count=500000)
    elif condition_type == "icd9":
        set_to_use = filter_condition_code_by_count(condition_code_to_count, min_count=0, max_count=500000)
    else:
        raise NotImplementedError()

    condition_code_to_index: Dict[str, int] = dict(zip(set_to_use, range(len(set_to_use))))

    mean_differential_sim, max_differential_sim, all_pair_differential_sim = [], [], []

    ## For each patient and condition, get a template, pass through BERT and return similarities

    for subject_id, patient_info in tqdm(subject_id_to_patient_info.items()):
        templates = []
        for condition in set_to_use:
            desc = condition_code_to_description[condition]
            templates.append(
                generate_template(patient_info.FIRST_NAME, patient_info.LAST_NAME, patient_info.GENDER, desc)
            )

        name = patient_info.FIRST_NAME + " " + patient_info.LAST_NAME
        name_length = len(tokenizer.tokenize(name))

        ## Following info may change if we change the template structure.
        ## Following are on basis of structure [CLS] {title} {name} is a yo patient with {condition} [SEP]
        example_template = tokenizer.tokenize(templates[0])
        name_start_index = 2  # Name Starts after [CLS] {title}
        name_end_index = name_start_index + name_length
        condition_start_index = example_template.index("patient") + 2
        condition_end_index = -1

        assert (
            tokenizer.convert_tokens_to_string(example_template[name_start_index:name_end_index])
            == " ".join(name.lower().split())
        ), breakpoint()
        assert (
            tokenizer.convert_tokens_to_string(example_template[condition_start_index:condition_end_index])
            == " ".join(condition_code_to_description[set_to_use[0]].lower().split())
        ), breakpoint()

        ## Pass all templates to BERT and return similarities

        mean_similarities, max_similarities, all_pair_similarities = get_name_condition_similarities(
            model,
            tokenizer,
            templates,
            name_start_index,
            name_end_index,
            condition_start_index,
            condition_end_index,
        )

        condition_labels = get_condition_labels_as_vector(patient_info.CONDITIONS, condition_code_to_index)

        mean_differential_sim.append(differential_similarity(condition_labels, mean_similarities))
        max_differential_sim.append(differential_similarity(condition_labels, max_similarities))
        all_pair_differential_sim.append(differential_similarity(condition_labels, all_pair_similarities))

    print(f"Mean Mean Pos-Neg {np.average(mean_differential_sim)}")
    print(f"SD Mean Pos-Neg {np.std(mean_differential_sim)}")
    print(f"Mean Max Pos-Neg {np.average(max_differential_sim)}")
    print(f"SD Max Pos-Neg {np.std(max_differential_sim)}")
    print(f"Mean All Pair Pos-Neg {np.average(all_pair_differential_sim)}")
    print(f"SD All Pair Pos-Neg {np.std(all_pair_differential_sim)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Location of the model", type=str, required=True)
    parser.add_argument("--tokenizer", help="Location of the tokenizer", type=str, required=True)
    parser.add_argument("--condition-type", type=str, choices=["icd9", "stanza"], required=True)

    args = parser.parse_args()
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)
    model = BertModel.from_pretrained(args.model).eval().cuda()
    main(model, tokenizer, args.condition_type)
