import torch

def noise_distribution(counts: torch.Tensor, alpha: float = 0.75) -> torch.Tensor:
    """
    Returns a torch.Tensor of shape (vocab_size,) containing the
    negative sampling noise distribution.

    P(w) = count(w)^alpha / sum(count(w)^alpha)
    """
    # Convert to floating point
    counts = counts.float()

    # Raise counts to the power alpha
    weights = counts.pow(alpha)

    # Normalize to obtain a probability distribution
    probs = weights / weights.sum()

    return probs