import argparse, os
import decoding
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from tqdm import tqdm

from utils import *


def eval_one(
    model: decoding.Base,
    tokenizer: PreTrainedTokenizer,
    question: dict,
):
    all_tokens = torch.empty((1, 0), dtype=torch.long, device=model.device)
    cache = decoding.DynamicCache()
    stat = decoding.Stat()

    tpl = chat_template[model.model_type]

    for i, turn in enumerate(question["turns"]):
        if i != 0:
            assert (
                cache.get_seq_length() + 1 == all_tokens.shape[1]
            ), f"{cache.get_seq_length()} + 1 != {all_tokens.shape[1]}"

        # Input
        messages = []
        if i == 0:  # And system message for first-time chat
            messages.append(tpl["sys"])
        messages.append(tpl["usr"].format(content=turn))  # User message for this turn
        messages.append(tpl["gen"])
        text = "".join(messages)
        turn_tokens = tokenizer([text], return_tensors="pt").input_ids.to(model.device)

        # Forward
        all_tokens = torch.cat((all_tokens, turn_tokens), dim=-1)
        with stat.tik_tok("generate"):
            output = model.generate(
                all_tokens, max_new_tokens=args.max_new_tokens, cache=cache, stat=stat
            )

        # Output
        cache = output["cache"]
        stat = output["stat"]
        all_tokens = output["output_ids"]

    # For debug
    # response = tokenizer.batch_decode(all_tokens, skip_special_tokens=False)[0]
    # print(response)

    return stat


def main(args):
    # Output
    os.makedirs("output", exist_ok=True)
    if args.output is None:
        args.output = f"output/{real_basename(args.model)}-{args.decode}.json"

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = apply_dec(model, args.decode)

    # Eval questions
    questions = load_jsonl(args.bench)
    stats = []
    for question in tqdm(questions):
        stat = eval_one(model, tokenizer, question)

        stat = stat.to_dict()
        stat["question_id"] = question["question_id"]
        stats.append(stat)

    # Dump stats
    save_jsonl(stats, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="./models/Qwen2.5-0.5B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "-d",
        "--decode",
        type=str,
        choices=["hf", "ar", "pld", "cyc"],
        default="ar",
        help="Decoding mode",
    )
    parser.add_argument(
        "--max-new-tokens", type=str, default=1024, help="Max new tokens"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")

    parser.add_argument(
        "-b",
        "--bench",
        type=str,
        default="./spec_bench/question.jsonl",
        help="Bench file",
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file")
    args = parser.parse_args()

    main(args)
