import numpy as np

def precision_at_k(labels, logits, k=10) -> float:
    top_k = np.argsort(logits)[-k:]
    top_k_labels = np.array(labels)[top_k]
    return top_k_labels.mean()

def differential_score(labels, scores) -> float:
    """Return average scores for positive class examples 
            - average scores for negative class examples
    """
    labels = np.array(labels)
    scores = np.array(scores)
    positive_scores = scores[labels == 1].mean()
    negative_scores = scores[labels == 0].mean()

    return positive_scores - negative_scores