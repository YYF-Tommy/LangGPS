import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import jsonlines
from tqdm.auto import tqdm
import time
import numpy as np
from collections import Counter, defaultdict
import argparse
from concurrent.futures import ThreadPoolExecutor


def template(model_name, input, output):
    if "Llama-3" in model_name:
        print("Apply Llama3 template !")
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}"
    elif "Qwen2.5" in model_name:
        print("Apply Qwen2.5 template !")
        return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{output}"


def get_dataset(path):
    data = []
    langs = []
    langs_ids = defaultdict(list)
    with jsonlines.open(path) as f:
        for i, line in enumerate(f):
            data.append({"instruction": line["inputs"], "output": line["targets"]})
            langs.append(line["language"])
            langs_ids[line["language"]].append(i)

    mapping = {}
    count = 0
    sorted_langs = sorted(set(langs))
    for sorted_lang in sorted_langs:
        for i in langs_ids[sorted_lang]:
            mapping[count] = i
            count += 1
    
    json.dump(mapping, open(f"{save_path}/mapping2vanilla.json", "w"))

    
    return langs, data

def batchify(data, batch_size=1):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

if __name__ == "__main__":
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    model_name = args.model_path
    data_path = args.data_path
    save_path = args.save_path

    device = "cuda"

    # where your full dataset is, please follow the format of './data/full.json', at least the three keys of "inputs", "targets", "language" should be kept.
    # data_path = "XXX"
    # save_path = "XXX" # where you want to save the vectors

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    langs, data = get_dataset(data_path)

    # data_langs = defaultdict(list)

    # for lang, item in tqdm(zip(langs, data)):
    #     data_langs[lang].append(template(model_name, item["instruction"], item["output"]))

    # time1 = time.time()

    # with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    #     print(data_langs) 
    #     for lang, data in tqdm(data_langs.items()):
    #         data = batchify(data)
    #         os.makedirs(f"{save_path}/{lang}", exist_ok=True)
    #         for item in tqdm(data):
    #             inputs = tokenizer(item, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    #             outputs = model(inputs.input_ids.to(device), output_hidden_states=True)
    #             hidden_states = outputs.hidden_states
    #             for idx in range(hidden_states[-1].shape[0]):
    #                 count = len(os.listdir(f"{save_path}/{lang}"))
    #                 np.save(f"{save_path}/{lang}/{count}.npy", hidden_states[-1][idx, -1].detach().clone().cpu().numpy())

    # time2 = time.time()

    # print(time2 - time1)



    # ### Stack the vectors

    # # Define the function to process each folder
    # def process_folder(folder):
    #     print(f"Processing folder: {folder}")
    #     files = os.listdir(f"{save_path}/{folder}")
    #     files.sort()
    #     cached_vectors = []
    #     for file in files:
    #         tmp = np.load(f"{save_path}/{folder}/{file}")
    #         cached_vectors.append(tmp)
    #     cached_vectors = np.stack(cached_vectors)

    #     print(cached_vectors.shape)

    #     # save_path = f"{save_path}"

        
    #     np.save(f"{save_path}/0_stacked/{folder}.npy", cached_vectors)

    
    # # List all the folders
    # folders = os.listdir(f"{save_path}")

    # os.makedirs(f"{save_path}/0_stacked", exist_ok=True)

    # # Use ThreadPoolExecutor to run the process_folder function for each folder concurrently
    # with ThreadPoolExecutor() as executor:
    #     executor.map(process_folder, folders)