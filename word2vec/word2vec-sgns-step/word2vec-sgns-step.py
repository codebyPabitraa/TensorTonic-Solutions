import torch

def sgns_sgd_step(
    W_in: torch.Tensor,
    W_out: torch.Tensor,
    center_id: int,
    pos_id: int,
    neg_ids: torch.Tensor,
    lr: float,
) -> tuple:
    """
    Returns tuple (W_in_updated, W_out_updated), each the same shape
    as the inputs, after one SGNS SGD step.
    """
    # Work on copies so the originals are not modified
    W_in_updated = W_in.clone()
    W_out_updated = W_out.clone()

    # Pre-update embeddings
    v_c = W_in[center_id].clone()
    u_o = W_out[pos_id].clone()
    u_negs = W_out[neg_ids].clone()  # (k, d)

    # Scores
    s_o = torch.dot(v_c, u_o)
    s_negs = u_negs @ v_c  # (k,)

    # Sigmoids
    sigma_o = torch.sigmoid(s_o)
    sigma_negs = torch.sigmoid(s_negs)

    # Output gradients
    grad_u_o = (sigma_o - 1.0) * v_c
    grad_u_negs = sigma_negs.unsqueeze(1) * v_c.unsqueeze(0)

    # Center gradient (using pre-update vectors)
    grad_v_c = (sigma_o - 1.0) * u_o + (sigma_negs.unsqueeze(1) * u_negs).sum(dim=0)

    # Apply SGD updates
    W_in_updated[center_id] -= lr * grad_v_c
    W_out_updated[pos_id] -= lr * grad_u_o

    for i, neg_id in enumerate(neg_ids):
        W_out_updated[neg_id] -= lr * grad_u_negs[i]

    return W_in_updated, W_out_updated