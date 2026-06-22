import torch
import torch.nn.functional as F

def cbow_forward(
    context_ids: torch.Tensor,
    target_id: int,
    W_in: torch.Tensor,
    W_out: torch.Tensor
) -> torch.Tensor:
    """
    Returns a scalar torch.Tensor: the CBOW cross-entropy loss for
    predicting target_id from the averaged context.
    """
    # Average the input embeddings of the context words
    h = W_in[context_ids].mean(dim=0)  # (embedding_dim,)

    # Compute logits for all vocabulary words
    logits = W_out @ h  # (vocab_size,)

    # Compute full-softmax cross-entropy loss
    loss = F.cross_entropy(
        logits.unsqueeze(0),                # (1, vocab_size)
        torch.tensor([target_id], device=logits.device)
    )

    return loss