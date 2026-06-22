import torch
import torch.nn.functional as F

def sgns_loss(center_vec: torch.Tensor,
              pos_vec: torch.Tensor,
              neg_vecs: torch.Tensor) -> torch.Tensor:
    """
    Computes the Skip-gram Negative Sampling (SGNS) loss.

    Args:
        center_vec: Tensor of shape (d,)
        pos_vec: Tensor of shape (d,)
        neg_vecs: Tensor of shape (k, d)

    Returns:
        Scalar torch.Tensor containing the SGNS loss.
    """
    # Positive score: v_c · u_o
    pos_score = torch.dot(center_vec, pos_vec)

    # Positive loss: -log(sigmoid(pos_score))
    pos_loss = F.softplus(-pos_score)

    # Negative scores: u_n · v_c for each negative sample
    neg_scores = neg_vecs @ center_vec  # shape: (k,)

    # Negative loss: -log(sigmoid(-score)) = softplus(score)
    neg_loss = F.softplus(neg_scores).sum()

    return pos_loss + neg_loss