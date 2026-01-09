import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from tqdm.auto import tqdm
import os
import time
import torch
import argparse
import json


if __name__ == "__main__":
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    save_path = args.save_path

    vectors = []
    labels = []
    files = os.listdir(f"{save_path}/0_stacked")
    files = [file for file in files if file.endswith(".npy")]
    files.sort()
    num_dict = {}
    language_cover = {}
    mapping = {}
    count = 0
    for file in tqdm(files):
        tmp = np.load(f"{save_path}/0_stacked/{file}")
        lang = file.replace(".npy", "")
        num_dict[lang] = len(tmp)
        language_cover[lang] = [count, count + len(tmp)]
        count += len(tmp)
        for i in range(len(tmp)):
            vectors.append(tmp[i])
            labels.append(lang)

    # 转换为 CuPy 数组
    data_matrix = cp.asarray(vectors)
    N = data_matrix.shape[0]

    # 计算距离矩阵（基于欧几里得距离）
    def compute_distance_matrix_gpu(data_matrix):
        norms = cp.sum(data_matrix**2, axis=1, keepdims=True)
        distance_matrix = cp.sqrt(cp.maximum(norms - 2 * cp.dot(data_matrix, data_matrix.T) + norms.T, 0))
        return distance_matrix

    # 计算 GPU 加速的距离矩阵
    dist_matrix = compute_distance_matrix_gpu(data_matrix)

    X = cp.asnumpy(dist_matrix)

    # 转换回 NumPy 并保存
    np.save(f"{save_path}/distance_matrix.npy", X)
    json.dump(language_cover, open(f"{save_path}/language_cover.json", "w"))


    # print(dist_matrix)

    # Scoring
    labels = np.array(labels)

    # X = dist_matrix

    np.fill_diagonal(X, 0)

    # 计算 Silhouette 分数
    silhouette_avg = silhouette_score(X, labels, metric="precomputed")
    silhouette_values = silhouette_samples(X, labels, metric="precomputed")

    np.save(f"{save_path}/silhouette_values.npy", silhouette_values)
    np.save(f"{save_path}/silhouette_avg.npy", silhouette_avg)

