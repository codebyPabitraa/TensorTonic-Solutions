import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns int64 torch.Tensor of shape (num_pairs, 2).
    """
    pairs = []
    n = token_ids.size(0)

    for i in range(n):
        start = max(0, i - window)
        end = min(n - 1, i + window)

        for j in range(start, end + 1):
            if j != i:
                pairs.append([token_ids[i].item(), token_ids[j].item()])

    if len(pairs) == 0:
        return torch.empty((0, 2), dtype=torch.int64)

    return torch.tensor(pairs, dtype=torch.int64)