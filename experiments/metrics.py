import numpy as np

def precision_at_k(labels, logits, k=10):
    top_k = np.argsort(logits)[-k:]
    top_k_labels = np.array(labels)[top_k]
    return top_k_labels.mean()

def differential_similarity(labels, similarity):
    labels = np.array(labels)
    similarity = np.array(similarity)
    positive_similarity = similarity[labels == 1].mean()
    negative_similarity = similarity[labels == 0].mean()

    return positive_similarity - negative_similarity