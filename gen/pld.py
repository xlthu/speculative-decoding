import torch
from transformers import DynamicCache

from .base import Generator

__all__ = ["PLD"]


class PLD(Generator):
    def __init__(self, model, max_matching_ngram_size=2, prompt_lookup_num_tokens=3):
        super().__init__(model)
        self.max_matching_ngram_size = max_matching_ngram_size
        self.prompt_lookup_num_tokens = prompt_lookup_num_tokens

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, **kwargs):
        batch_size, input_len = input_ids.shape
        assert batch_size == 1, "batch_size must be 1"
        n_ctx = input_len + max_new_tokens

        cache = DynamicCache()
        tokens = torch.clone(input_ids)
        while True:
            n_past = cache.get_seq_length()
            print(f"=== {n_past} ===")

            # Draft
            draft = self.find_candidate_pred_tokens(tokens)  # [n_draft]
            n_draft = draft.shape[0]

            # Input
            cur_tokens = torch.cat((tokens[:, n_past:], draft.unsqueeze(0)), dim=1)
            n_token = cur_tokens.shape[1]
            attention_mask = self.prepare_attention_mask(n_past, n_token)
            position_ids = self.prepare_position_ids(n_past, n_token)

            # Verify
            output = self.model(
                input_ids=cur_tokens,
                attention_mask=None,
                position_ids=None,
                past_key_values=cache,
                return_dict=True,
                logits_to_keep=1 + n_draft,
            )

            # Output
            logits = output.logits.squeeze(0)  # [n_draft + 1, n_vocab]
            gen_ids = torch.argmax(logits, dim=-1)  # [n_draft + 1]

            acc_len = 1
            for acc_len in range(1, gen_ids.shape[0]):
                if gen_ids[acc_len - 1] != draft[acc_len - 1]:
                    break

            print(f"{acc_len=}")
            gen_ids = gen_ids[:acc_len].unsqueeze(0)  # [1, acc_len]
            print(f"{gen_ids=}")
            tokens = torch.cat((tokens, gen_ids), dim=-1)

            if self.is_finished(tokens, n_ctx, gen_ids):
                return tokens

            # Update
            cache: DynamicCache = output.past_key_values
            # print(cache.get_seq_length())
            cache.crop(tokens.shape[1] - 1)
            # print(cache.get_seq_length())

    def is_eos(self, token: int):
        return token in self.get_eos_token_ids()

    def is_finished(self, tokens: torch.Tensor, n_ctx: int, gen_ids: torch.Tensor):
        if tokens.shape[1] >= n_ctx:
            return True

        for token in gen_ids[0]:
            if self.is_eos(token):
                return True

        return False

    def prepare_attention_mask(self, n_past: int, n_token: int):
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
        pos = torch.arange(n_past, n_token + n_past, device=self.device)
        pos.unsqueeze_(0)
        return pos

    def reserve_until_eos(self, tokens: torch.Tensor):
        # tokens: [n]
        tokens_list = tokens.tolist()
        for i, token in enumerate(tokens_list):
            if self.is_eos(token):
                return tokens[: i + 1]
        return tokens

    def find_candidate_pred_tokens(self, tokens: torch.Tensor):
        # tokens: [batch_size, n_past + n_token]
        # return draft: [n_draft]
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
                    return self.reserve_until_eos(tokens[0, start_idx:end_idx])

        # If no match is found, return an empty tensor
        return torch.tensor([], dtype=torch.long, device=tokens.device)
