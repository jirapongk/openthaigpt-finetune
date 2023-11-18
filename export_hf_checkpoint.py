import os
import sys

import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402

if len(sys.argv) != 3:
    print("Usage: python export_hf_checkpoint.py <base_model> <lora_dir>")
    exit()
# BASE_MODEL = os.environ.get("BASE_MODEL", None)
# BASE_MODEL = 'openthaigpt/openthaigpt-1.0.0-beta-7b-chat-ckpt-hf'
BASE_MODEL = sys.argv[1]
# lora_dir = './openthaigpt-100-beta-7b-bella-casedetail'
lora_dir = sys.argv[2]
assert (
    BASE_MODEL
), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=decapoda-research/llama-7b-hf`"  # noqa: E501

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

base_model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    lora_dir,
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

lora_weight = lora_model.base_model.model.model.layers[
    0
].self_attn.q_proj.weight

assert torch.allclose(first_weight_old, first_weight)

# merge weights - new merging method from peft
lora_model = lora_model.merge_and_unload()

lora_model.train(False)

# did we do anything?
assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

LlamaForCausalLM.save_pretrained(
    base_model, "./hf_ckpt", state_dict=deloreanized_sd, max_shard_size="400MB"
)
