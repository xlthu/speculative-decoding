import torch
from transformers import DynamicCache

from .base import Base

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

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, **kwargs):
        n_batch, n_input = input_ids.shape
        assert n_batch == 1, "batch size must be 1"
        n_ctx = n_input + max_new_tokens

        cache = DynamicCache()
        all_tokens = torch.clone(input_ids)  # [1, n_past + n_token]
        while True:
            n_past = cache.get_seq_length()
            print(f"=== {n_past} ===")

            # Draft
            draft = self.draft(all_tokens)  # [1, n_draft]
            n_draft = draft.shape[1]

            # Input
            in_tokens = torch.cat((all_tokens[:, n_past:], draft), dim=-1)
            n_token = in_tokens.shape[1]  # [1, n_token]
            attention_mask = self.prepare_attention_mask(n_past, n_token)
            position_ids = self.prepare_position_ids(n_past, n_token)

            # Forward
            output = self.model(
                input_ids=in_tokens,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                return_dict=True,
                logits_to_keep=1 + n_draft,
            )

            # Output
            out_tokens = torch.argmax(output.logits, dim=-1)  # [1, n_draft + 1]

            # Verify
            acc_len = 1
            for acc_len in range(1, 1 + n_draft):
                if out_tokens[0, acc_len - 1] != in_tokens[0, acc_len]:
                    break

            out_tokens = out_tokens[:, :acc_len]  # [1, acc_len]
            all_tokens = torch.cat((all_tokens, out_tokens), dim=-1)

            # Update
            cache: DynamicCache = output.past_key_values
            cache.crop(all_tokens.shape[1] - 1)

            if self.is_finished(all_tokens, n_ctx, out_tokens):
                return all_tokens

    def prepare_attention_mask(self, n_past: int, n_token: int):
        # return attention_mask: [1, 1, n_token, n_past + n_token]
        # 0: allow attention, -inf: not allow attention
        min_dtype = torch.finfo(self.dtype).min
        mask = torch.full(
            (n_token, n_past + n_token),
            fill_value=min_dtype,
            dtype=self.dtype,
            device=self.device,
        )
        mask = torch.triu(mask, diagonal=n_past + 1)
        mask = mask.reshape(1, 1, *mask.shape)

        return mask

    def prepare_position_ids(self, n_past: int, n_token: int):
        # return position_ids: [1, n_token]
        pos = torch.arange(n_past, n_token + n_past, device=self.device)
        pos.unsqueeze_(0)
        return pos

    def reserve_until_eos(self, tokens: torch.Tensor):
        # tokens: [1, n]
        for i, token in enumerate(tokens.flatten()):
            if self.is_eos(token):
                return tokens[:, : i + 1]
        return tokens

    def draft(self, tokens: torch.Tensor):
        # tokens: [1, n_past + n_token]
        # return draft: [1, n_draft]
        input_length = tokens.shape[1]

        for ngram_size in range(self.max_matching_ngram_size, 0, -1):
            # Extract the last n tokens as our search ngram
            ngram = tokens[0, -ngram_size:].unsqueeze(0)

            # Create sliding windows of size ngram_size
            windows = tokens.unfold(dimension=1, size=ngram_size, step=1)

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
                    return self.reserve_until_eos(tokens[:, start_idx:end_idx])

        # If no match is found, return an empty tensor
        return torch.tensor([[]], dtype=torch.long, device=tokens.device)
