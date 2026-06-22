import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns a torch.Tensor of shape (vocab_size,) containing the
    subsampling keep probability for each word.

    P_keep(w) = min(1, sqrt(t / f(w)))
    where f(w) = count(w) / total_count.
    """
    counts = counts.float()
    total_count = counts.sum()

    # Compute word frequencies
    freqs = counts / total_count

    # Compute keep probabilities
    keep_probs = torch.sqrt(t / freqs)

    # Cap probabilities at 1
    keep_probs = torch.minimum(
        keep_probs,
        torch.ones_like(keep_probs)
    )

    return keep_probs