import argparse
import decoding
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer


def apply_dec(model, args):
    match args.dec:
        case "hf":
            return decoding.HF(model)
        case "ar":
            return decoding.AutoRegressive(model)
        case "pld":
            return decoding.PLD(model)
        case "cyc":
            return decoding.Recycle(model)
        case _:
            assert ValueError(args.dec)

    return None


def gen_one(
    model: decoding.Base,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
):
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=128)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    print(generated_ids)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = apply_dec(model, args)

    response = gen_one(model, tokenizer, args.prompt)

    print(response)


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
        "--dec",
        type=str,
        choices=["hf", "ar", "pld", "cyc"],
        default="ar",
        help="Decoding mode",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="Give me a short introduction to large language model.",
        help="Prompt",
    )
    args = parser.parse_args()

    main(args)
