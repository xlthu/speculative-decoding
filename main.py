import argparse
import decoding
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from utils import *


def gen_one(
    model: decoding.Base,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
):
    tpl = chat_template[model.model_type]
    messages = [
        tpl["sys"],
        tpl["usr"].format(content=prompt),
        tpl["gen"],
    ]
    text = "".join(messages)
    print(text)
    input_ids = tokenizer([text], return_tensors="pt").input_ids.to(model.device)

    output = model.generate(input_ids, max_new_tokens=args.max_new_tokens)

    generated_ids = output["output_ids"]
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(input_ids, generated_ids)
    ]
    print(f"{generated_ids=}")
    print(f"{output['stat']=}")

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = apply_dec(model, args.decode, args.eagle)

    response = gen_one(model, tokenizer, args.prompt)

    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="./models/Qwen2-7B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--eagle",
        type=str,
        default="./models/EAGLE-Qwen2-7B-Instruct",
        help="Eagle model path",
    )
    parser.add_argument(
        "-d",
        "--decode",
        type=str,
        choices=["hf", "ar", "pld", "cyc", "ea"],
        default="ar",
        help="Decoding mode",
    )
    parser.add_argument(
        "--max-new-tokens", type=str, default=1024, help="Max new tokens"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="Give me a short introduction to large language model.",
        help="Prompt",
    )
    args = parser.parse_args()

    main(args)
