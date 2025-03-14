import os, json
import decoding
import decoding.eagle_model

__all__ = [
    "load_json",
    "save_json",
    "load_jsonl",
    "save_jsonl",
    "real_basename",
    "apply_dec",
    "chat_template",
]


def load_json(json_path: str):
    with open(json_path, "r") as f:
        return json.load(f)


def save_json(obj, json_path: str):
    with open(json_path, "w") as f:
        json.dump(obj, f)


def load_jsonl(jsonl_path: str):
    ret = []
    with open(jsonl_path, "r") as f:
        for line in f:
            ret.append(json.loads(line))
    return ret


def save_jsonl(obj_list: list, jsonl_path: str):
    with open(jsonl_path, "w") as f:
        for obj in obj_list:
            json.dump(obj, f)
            f.write("\n")


def real_basename(path: str):
    if path.endswith(os.sep):
        path = path.removesuffix(os.sep)
    return os.path.basename(path)


def apply_dec(model, dec_type: str, eagle_path: str):
    match dec_type:
        case "hf":
            return decoding.HF(model)
        case "ar":
            return decoding.AutoRegressive(model)
        case "pld":
            return decoding.PLD(model)
        case "cyc":
            return decoding.Recycle(model, model.config.vocab_size)
        case "ea":
            ea = decoding.eagle_model.EAModel.from_pretrained(eagle_path)
            ea = ea.to(model.device)
            return decoding.Eagle(model, ea, h=2, k=2, m=4)
        case _:
            assert ValueError(dec_type)

    return None


chat_template = {}

chat_template["qwen2"] = {
    "sys": """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>""",
    "usr": """
<|im_start|>user
{content}<|im_end|>""",
    "gen": """
<|im_start|>assistant
""",
}
