import sys

import os
import json
import jsonlines
import numpy as np
import argparse
from collections import defaultdict


save_langs = ["Basque", "Bengali", "Egyptian Arabic", "English", "French", "German", "Greek", "Haitian", "Hindi", "Indonesian", "Italian", "Japanese", "Korean", "Moroccan Arabic", "Najdi Arabic", "Portuguese", "Russian", "Simplified Chinese", "South Levantine Arabic", "Spanish", "Standard Arabic", "Swahili", "Ta'izzi-Adeni Arabic", "Tamil", "Telugu", "Thai", "Turkish", "Ukrainian", "Urdu", "Vietnamese", "Yoruba"]

def top_indices(lst, num):
    return np.argsort(lst)[-num:][::-1]
    # return np.argsort(lst)[:num]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--percent", type=int)
    args = parser.parse_args()

    data_path = args.data_path
    save_path = args.save_path
    percent = args.percent

    # 1% = 977
    # 3% = 2931
    # 5% = 4885

    # 48848
    # percent = 20

    silhouette_scores = np.load(f"{save_path}/silhouette_values.npy")
    # silhouette_scores_avg = np.load(f"{save_path}/silhouette_avg.npy")

    print(len(silhouette_scores))

    language_cover = json.load(open(f"{save_path}/language_cover.json", "r"))
    langs = list(language_cover.keys())
    langs.sort()
    print(langs)

    selected_num = {}
    selected_ids = []
    # selected_ids = defaultdict(list)
    for lang in langs:
        cover = language_cover[lang]
        selected_num[lang] = round((cover[1] - cover[0]) * 0.01 * percent)


    total = sum(selected_num.values())
    diff = total - round(len(silhouette_scores) * 0.01 * percent)

    max_key = max(selected_num, key=selected_num.get)
    selected_num[max_key] -= diff

    for lang in langs:
        # if lang == "English":
        num = selected_num[lang]
        cover = language_cover[lang]
        tmp = [item+cover[0] for item in top_indices(silhouette_scores[cover[0]: cover[1]], num)]
        # selected_list.extend(tmp)
        # selected_ids[lang] = tmp
        selected_ids.extend(tmp)
    selected_ids.sort()

    mapping2vanilla = json.load(open(f"{save_path}/mapping2vanilla.json", "r"))

    selected_ids_in_vanilla = [mapping2vanilla[str(item)] for item in selected_ids]

    selected_data = []
    with jsonlines.open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i in selected_ids_in_vanilla:
                selected_data.append({"instruction": line["inputs"], "output": line["targets"]})

    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/selected_top_{percent}%.json", 'w', encoding='utf-8') as f:
        json.dump(saved_data, f, ensure_ascii=False, indent=2)
