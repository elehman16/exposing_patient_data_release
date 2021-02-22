import argparse
from typing import Dict

import gensim
import numpy as np
import spacy
from experiments.utilities import (
    filter_condition_code_by_count,
    get_condition_code_to_count,
    get_condition_code_to_descriptions,
    get_condition_labels_as_vector,
    get_subject_id_to_patient_info,
)
from experiments.metrics import differential_score
from tqdm import tqdm

normalize = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

# load tokenizer
nlp = spacy.load("en_core_sci_sm")
tokenizer = nlp.Defaults.create_tokenizer(nlp)


def get_embedding(model: gensim.models.KeyedVectors, text: str):
    mean_vector = model.vectors.mean(0)[None, :]  ## Hackery since using full model not trained yet
    vectors = [model[token.text.lower()][None, :] for token in tokenizer(text) if token.text.lower() in model]
    if len(vectors) > 0:
        return np.concatenate(
            vectors,
            axis=0,
        )
    else:
        return np.zeros_like(mean_vector)


def main(model: gensim.models.KeyedVectors, condition_type: str):
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

    mean_condition_embeddings = []
    max_condition_embeddings = []
    all_condition_embeddings = []

    for condition in set_to_use:
        desc = condition_code_to_description[condition]
        condition_embeddings = get_embedding(model, desc)

        mean_embedding = normalize(np.mean(condition_embeddings, axis=0, keepdims=True))
        mean_condition_embeddings.append(mean_embedding)

        max_embedding = normalize(np.max(condition_embeddings, axis=0, keepdims=True))
        max_condition_embeddings.append(max_embedding)

        all_condition_embeddings.append(normalize(condition_embeddings))

    mean_condition_embeddings = np.concatenate(
        mean_condition_embeddings, axis=0
    )  # Shape = (Num_Conditions, Embedding Size)
    max_condition_embeddings = np.concatenate(
        max_condition_embeddings, axis=0
    )  # Shape = (Num_Conditions, Embedding Size)

    mean_differential_sim, max_differential_sim, all_pair_differential_sim = [], [], []

    for subject_id, patient_info in tqdm(subject_id_to_patient_info.items()):
        name = patient_info.FIRST_NAME + " " + patient_info.LAST_NAME
        name_embeddings = get_embedding(model, name)

        mean_name_embedding = normalize(name_embeddings.mean(0))  # Shape = (Embedding Size,)
        max_name_embedding = normalize(name_embeddings.max(0))  # Shape = (Embedding Size,)

        mean_similarities = mean_condition_embeddings @ mean_name_embedding
        max_similarities = max_condition_embeddings @ max_name_embedding

        name_embeddings = normalize(name_embeddings)

        all_pair_similarities = []
        for condition_embeddings in all_condition_embeddings :
            similarity_matrix = condition_embeddings @ name_embeddings.T
            all_pair_similarities.append(np.max(similarity_matrix))

        condition_labels = get_condition_labels_as_vector(patient_info.CONDITIONS, condition_code_to_index)

        mean_differential_sim.append(differential_score(condition_labels, mean_similarities))
        max_differential_sim.append(differential_score(condition_labels, max_similarities))
        all_pair_differential_sim.append(differential_score(condition_labels, all_pair_similarities))

    print(f"Mean Mean Pos-Neg {np.average(mean_differential_sim)}")
    print(f"SD Mean Pos-Neg {np.std(mean_differential_sim)}")
    print(f"Mean Max Pos-Neg {np.average(max_differential_sim)}")
    print(f"SD Max Pos-Neg {np.std(max_differential_sim)}")
    print(f"Mean All Pair Pos-Neg {np.average(all_pair_differential_sim)}")
    print(f"SD All Pair Pos-Neg {np.std(all_pair_differential_sim)}")


from training_scripts.train_word_embeddings import callback as callback

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", help="Location of the model.", type=str)
    parser.add_argument("--condition-type", type=str, choices=["icd9", "stanza"], required=True)

    args = parser.parse_args()
    model = gensim.models.Word2Vec.load(args.model_file).wv
    main(model, args.condition_type)
