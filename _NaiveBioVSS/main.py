import os
import sys

import time
import numpy as np
import hausdorff_distance_naive_lsh
import cal_distance

import torch

recall_all = {3: [], 5: [], 10: [], 15: [], 20: [], 25: [], 30: []}
def dingzhi_compare_benchmark(real_distances, search_result):
    """
    比较基准数据集的结果。
    """
    # 使用 torch.topk 找到前 k 个最小值及其索引
    topk_values, real = torch.topk(real_distances, 30, largest=False)
    search = torch.tensor(search_result)

    for topK in recall_all.keys():
        recall_all[topK].append(len(set(real[:topK].tolist()) & set(search.tolist())) / topK)

    # for topK in recall_all.keys():
        # real = list(sorted(range(len(real_distances)), key=lambda i: real_distances[i]))
        # search = [item for item in search_result]
        # recall_all[topK].append(len(set(real[:topK]) & set(search[:topK])) / topK)


    return recall_all

class NaiveBioVSS:
    def __init__(self, config, dataloader=None):
        self.config = config

        self.pickle_file_path = f'/data1/brucelee/AuthorBinaryData/{config["dataSet"]}/author_Bio_FLSH_64_1024_vectors.pickle'
        self.all_vectors, self.lengths, self.starts, self.vector_dim, self.num_sets = self.load_and_prepare_vectors()

        self.concatenated_matrix = dataloader.concat_dense_vector_matrix["concatenated_matrix"]
        self.cumulative_offsets = dataloader.concat_dense_vector_matrix["cumulative_offsets"]
        self.set_lengths = dataloader.concat_dense_vector_matrix["set_lengths"]
        self.dense_author_vectors = dataloader.dense_author_vectors

        # 计算平均向量用于构建倒排索引
        self.author_vectors_dense_mean = torch.stack(
            [torch.mean(author_vector, dim=0) for author_vector in self.dense_author_vectors])

        # 用于统计该方法平均召回率
        self.recall_all = {3: [], 5: [], 10: [], 15: [], 20: [], 25: [], 30: []}

        self.device = config["device"]

    def load_and_prepare_vectors(self):
        # 读取真实的向量集合数据
        with open(self.pickle_file_path, 'rb') as file:
            author_vectors = torch.load(file)

        print("作者向量集大小为:", len(author_vectors))

        # 获取向量集合的数量和向量的维度
        num_sets = len(author_vectors)
        vector_dim = author_vectors[0].size(1)
        n_words = (vector_dim + 63) // 64

        # 获取每个集合的长度
        lengths = np.array([len(vec_set) for vec_set in author_vectors], dtype=np.int64)

        # 计算每个集合的起始位置
        starts = np.zeros(num_sets, dtype=np.int64)
        current_start = 0
        for i in range(num_sets):
            starts[i] = current_start
            current_start += lengths[i]

        # 计算总向量数
        total_vectors = int(np.sum(lengths))

        # 生成打包向量
        all_vectors = np.zeros((total_vectors * n_words,), dtype=np.uint64)

        # 生成转换矩阵
        bit_matrix = (2 ** np.arange(64, dtype=np.uint64)).reshape(64, 1)

        index = 0
        print("开始生成")
        for vec_set in author_vectors:
            for vec in vec_set:
                bits_reshaped = vec.reshape(-1, 64)
                # 进行矩阵乘法并打包为uint64
                packed_vector = np.dot(bits_reshaped, bit_matrix).flatten()
                all_vectors[index * n_words : (index + 1) * n_words] = packed_vector
                index += 1

        return all_vectors, lengths, starts, vector_dim, num_sets

    def compute_hausdorff_distances(self, query_index, condidata=50000):
        print("查询的索引为：", query_index)
        start_time = time.time()
        hausdorff_distances = hausdorff_distance_naive_lsh.compute_hausdorff_distances(
            self.all_vectors, self.lengths, self.starts, query_index, self.vector_dim
        )
        hausdorff_distances = torch.from_numpy(hausdorff_distances)
        top_value, top_index = hausdorff_distances.topk(condidata, largest=False)
        end_time = time.time()
        print("时间为", end_time-start_time)
        return top_value, top_index

    def query(self, query_index, topK):
        # 第一步：通过索引来筛选候选集合
        top_value, search_result =self.compute_hausdorff_distances(query_index)

        # 第二步：根据候选点计算真实距离
        distances = self.cal_p2b_distance(query_index, search_result)
        # 第三步：根据真实距离排序，获取最终的索引
        sorted_indices = torch.argsort(distances)[:topK]

        return [search_result[r].item() for r in sorted_indices]

    def cal_p2b_distance(self, query_index, candidate_index_set):
        """计算单项对多项之间的距离"""
        return self._cal_p2b_distance(query_index, candidate_index_set)

    def _cal_p2b_distance(self, query_index, candidate_index_set):
        distances = cal_distance.cal_p2b_distance(
            query_index,
            candidate_index_set,
            self.dense_author_vectors[query_index],
            self.cumulative_offsets,
            self.set_lengths,
            self.concatenated_matrix,
            self.device
        )
        return distances

# 使用示例
# 日志在2024年7月4日的文件夹




# # 查询和计算召回率
# print("开始查询")
# for query_index, real_distances in zip(query_index_list, real_distances_list):
#     top_value, top_index = calculator.compute_hausdorff_distances(query_index)
#
#     # 计算召回率
#     recall_all = dingzhi_compare_benchmark(real_distances, top_index)
#
#     recall = {k: sum(v) / len(v) for k, v in recall_all.items()}
#     print(f"召回率为{recall}")


