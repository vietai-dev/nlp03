import argparse
import json

import torch
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, AutoConfig

from lora_model import LoraModelForCasualLM
from prompt import Prompter

def get_response(prompt, tokenizer, model, generation_config, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
                input_ids=inputs['input_ids'].cuda(),
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
                do_sample=True)
    output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return output

def generate_inference(instruction: str, user_inp: str, model_path:str, lora_weights_path:str):
    
    top_k=40
    top_p=128
    temperature=0.1
    load_8bit=True
    num_beams =1
    max_new_tokens =128

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    config = AutoConfig.from_pretrained(model_path)
    architecture = config.architectures[0]
    if "Llama" in architecture:
        print("Setting EOS, BOS, UNK, and PAD tokens for LLama tokenizer")
        tokenizer.add_special_tokens(
            {
                "eos_token": "</s>",
                "bos_token": "</s>",
                "unk_token": "</s>",
            }
        )
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto")
    model = LoraModelForCasualLM.from_pretrained(
            model,
            lora_weights_path,
            torch_dtype=torch.float16)
        
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.eval()
    
    generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams)
    
    prompter = Prompter()
    
   
    
    # instruction = input("Your Instruction: ")
    # user_inp = input("Your input (enter n/a if there is no): ")
    if user_inp.lower().strip() == "n/a":
        user_inp = None
    prompt = prompter.generate_prompt(instruction, user_inp)
    output = get_response(prompt, tokenizer, model, generation_config, max_new_tokens)
    response = prompter.get_response(output)
    return response
    
