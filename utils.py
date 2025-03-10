import json
import decoding

__all__ = ["load_json", "save_json", "load_jsonl", "save_jsonl", "apply_dec"]


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


def apply_dec(model, dec_type: str):
    match dec_type:
        case "hf":
            return decoding.HF(model)
        case "ar":
            return decoding.AutoRegressive(model)
        case "pld":
            return decoding.PLD(model)
        case "cyc":
            return decoding.Recycle(model, model.config.vocab_size)
        case _:
            assert ValueError(dec_type)

    return None
