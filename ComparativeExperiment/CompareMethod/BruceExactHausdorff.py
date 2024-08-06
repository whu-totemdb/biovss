import time

import torch

from scipy.spatial.distance import directed_hausdorff

import cal_distance

class BruceExactHausdorff():
    def __init__(self, dataloader = None, device = "cpu"):
        self.device = device
        assert dataloader is not None, "dataloader is None. 需要预先加载，为后续计算做准备"

        self.concatenated_matrix = dataloader.concat_dense_vector_matrix["concatenated_matrix"]
        self.cumulative_offsets = dataloader.concat_dense_vector_matrix["cumulative_offsets"]
        self.set_lengths = dataloader.concat_dense_vector_matrix["set_lengths"]
        self.dense_author_vectors = dataloader.dense_author_vectors

        # 用于暴力搜索的全部索引
        self.all_index = torch.arange(len(dataloader.dense_author_vectors))

        # 用于统计该方法平均召回率
        self.recall_all = {3: [], 5: [], 10: []}

    def query(self, query_index, topK):
        distances = self.cal_p2b_distance(query_index, self.all_index)
        # 第二步：根据真实距离排序，获取最终的索引
        sorted_indices = torch.argsort(distances)[:topK]

        return [self.all_index[r].item() for r in sorted_indices]

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

