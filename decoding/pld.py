import torch
from transformers import DynamicCache

from .base import Base
from .utils import chain_attention_mask, chain_position_ids

__all__ = ["PLD"]


class PLD(Base):
    def __init__(self, model, max_matching_ngram_size=2, prompt_lookup_num_tokens=3):
        """
        Args:
            max_matching_ngram_size (int, optional): The maximum ngram size to be considered
                for matching in the prompt. Defaults to 2.
            prompt_lookup_num_tokens (int, optional): The number of tokens to be output as
                candidate tokens. Defaults to 3.
        """
        super().__init__(model)
        self.max_matching_ngram_size = max_matching_ngram_size
        self.prompt_lookup_num_tokens = prompt_lookup_num_tokens

    ### Input

    def get_input(
        self, all_tokens: torch.Tensor, cache: DynamicCache
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_past = cache.get_seq_length()

        # Draft
        dchain = self.draft(all_tokens)  # [1, n_dr]

        # Input
        in_tokens = torch.cat((all_tokens[:, n_past:], dchain), dim=-1)
        n_token = in_tokens.shape[1]  # [1, n_token]

        attention_mask = chain_attention_mask(n_past, n_token, self.dtype, self.device)
        position_ids = chain_position_ids(n_past, n_token, self.device)

        # Record
        self.dchain = dchain

        return in_tokens, attention_mask, position_ids

    def reserve_until_eos(self, tokens: torch.Tensor):
        # tokens: [1, n]
        for i, token in enumerate(tokens.flatten()):
            if self.is_eos(token):
                return tokens[:, : i + 1]
        return tokens

    def draft(self, all_tokens: torch.Tensor):
        # all_tokens: [1, n_past + n_token]
        # return draft: [1, n_dr]
        input_length = all_tokens.shape[1]

        for ngram_size in range(self.max_matching_ngram_size, 0, -1):
            # Extract the last n tokens as our search ngram
            ngram = all_tokens[0, -ngram_size:].unsqueeze(0)

            # Create sliding windows of size ngram_size
            windows = all_tokens.unfold(dimension=1, size=ngram_size, step=1)

            # Find where the windows match the ngram
            matches = (windows == ngram).all(dim=2)

            # Get the indices of matches
            match_indices = matches.nonzero(as_tuple=True)[1].tolist()

            # Iterate through match indices to find a valid continuation
            for idx in reversed(match_indices):
                start_idx = idx + ngram_size
                end_idx = start_idx + self.prompt_lookup_num_tokens
                # Ensure we don't go beyond the length of tokens and avoid self-match
                if end_idx <= input_length and start_idx < input_length - ngram_size:
                    return self.reserve_until_eos(all_tokens[:, start_idx:end_idx])

        # If no match is found, return an empty tensor
        return torch.tensor([[]], dtype=torch.long, device=all_tokens.device)

    ### Output

    def obtain_output(
        self,
        in_tokens: torch.Tensor,
        logits: torch.Tensor,
        cache: DynamicCache,
    ) -> torch.Tensor:
        # Output
        n_dr = self.dchain.shape[1]
        out_tokens = torch.argmax(logits[:, -n_dr - 1 :, :], dim=-1)  # [1, 1 + n_dr]

        # Verify
        acc_len = 1  # Accept AR output
        if n_dr:  # Accept draft tokens before the first unequal position
            neq = in_tokens[0, -n_dr:] != out_tokens[0, :-1]
            acc_len += neq.int().argmax(-1).item()

        out_tokens = out_tokens[:, :acc_len]  # [1, acc_len]

        # Update
        if n_dr:
            self.update_kv_cache(cache, cache.get_seq_length() - n_dr + acc_len - 1)

        return out_tokens
