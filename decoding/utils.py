import torch

from .dtree import DraftTree

__all__ = [
    "chain_attention_mask",
    "chain_position_ids",
    "tree_attention_mask",
    "tree_position_ids",
]


def chain_attention_mask(n_past: int, n_token: int, dtype, device):
    # return attention_mask: [1, 1, n_token, n_past + n_token]
    # 0: allow attention, -inf: not allow attention
    min_dtype = torch.finfo(dtype).min
    mask = torch.full(
        (n_token, n_past + n_token),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    mask = torch.triu(mask, diagonal=n_past + 1)
    mask = mask.reshape(1, 1, *mask.shape)

    return mask


def chain_position_ids(n_past: int, n_token: int, device):
    # return position_ids: [1, n_token]
    return torch.arange(
        n_past, n_token + n_past, dtype=torch.long, device=device
    ).unsqueeze(0)


def tree_attention_mask(n_past: int, n_dc: int, dtree: DraftTree, dtype, device):
    # return attention_mask: [1, 1, n_dc + n_dr, n_past + n_dc + n_dr]
    # 0: allow attention, -inf: not allow attention
    n_dr = dtree.size()
    min_dtype = torch.finfo(dtype).min

    lmask = torch.full(
        size=(n_dc + n_dr, n_past + n_dc),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    lmask = torch.triu(lmask, diagonal=n_past + 1)

    rmask = torch.full(
        size=(n_dc + n_dr, n_dr),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    dr_mask = rmask[n_dc:, :]
    dtree.zero_mask(dr_mask)

    mask = torch.cat((lmask, rmask), dim=-1)
    mask = mask.reshape(1, 1, *mask.shape)

    return mask


def tree_position_ids(n_past: int, n_dc: int, dtree: DraftTree, device):
    # return position_ids: [1, n_dc + n_dr]
    dc_pos = torch.arange(
        n_past, n_past + n_dc, dtype=torch.long, device=device
    ).unsqueeze(0)
    dr_pos = dtree.position_ids(n_past + n_dc, device)
    pos = torch.cat((dc_pos, dr_pos), dim=-1)
    return pos
