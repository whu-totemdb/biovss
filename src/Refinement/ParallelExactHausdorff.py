import time

import torch

from scipy.spatial.distance import directed_hausdorff

import cal_distance

class ParallelExactHausdorff():
    def __init__(self, dataloader = None, device = "cpu"):
        self.device = device
        assert dataloader is not None, "dataloader is None. 需要预先加载，为后续计算做准备"

        start_time = time.time()
        self.concatenated_matrix = dataloader.concat_dense_vector_matrix["concatenated_matrix"]
        self.cumulative_offsets = dataloader.concat_dense_vector_matrix["cumulative_offsets"]
        self.set_lengths = dataloader.concat_dense_vector_matrix["set_lengths"]
        print(f"类之间的数据加载测试，耗时: {time.time() - start_time} s.")

    def cal_p2p_distance(self):
        """计算单项之间的距离"""
        pass

    def cal_p2b_distance(self, query_index, candidate_index_set, dataloader):
        """计算单项对多项之间的距离"""
        return self._cal_p2b_distance(query_index, candidate_index_set, dataloader)

    def cal_b2b_distance(self):
        """计算多项对多项的距离矩阵"""
        pass

    def _cal_p2b_distance(self, query_index, candidate_index_set, dataloader):
        distances = cal_distance.cal_p2b_distance(
            query_index,
            candidate_index_set,
            dataloader.dense_author_vectors[query_index],
            self.cumulative_offsets,
            self.set_lengths,
            self.concatenated_matrix,
            self.device
        )
        return distances


    # def _cal_p2b_distance(self, query_index, candidate_index_set, dataloader):
    #     # 从 dense_author_vectors 中获取查询向量的集合
    #     query_set = dataloader.dense_author_vectors[query_index].to(self.device)
    #
    #
    #     batch_size = 102400  # 定义批处理大小
    #
    #     # 存储 Hausdorff 距离的列表
    #     hausdorff_distances = torch.ones(len(candidate_index_set))
    #
    #     # 计算每个查询点到其候选集的最小距离
    #     for i in range(0, len(candidate_index_set), batch_size):
    #         print("时间点1:", time.time())
    #         # 获取当前批次的候选作者索引
    #         candidate_index_batch = candidate_index_set[i: i + batch_size]
    #
    #         cumulative_offsets_batch = self.cumulative_offsets[candidate_index_batch] # 获取当前的结束索引
    #         set_lengths_batch = self.set_lengths[candidate_index_batch] # 每个向量集的长度
    #
    #         # 获取当前批次的候选向量索引和对应的向量矩阵
    #         vectors_index_batch = torch.cat([torch.arange(start, end) for start, end in zip(cumulative_offsets_batch - set_lengths_batch, cumulative_offsets_batch)], dim=0)
    #         concatenated_matrix_batch = self.concatenated_matrix[vectors_index_batch]
    #
    #         print("时间点---:", time.time())
    #         # 计算查询集合与候选集合之间的距离矩阵
    #         dist_matrix = torch.cdist(query_set.unsqueeze(0), concatenated_matrix_batch.unsqueeze(0)).squeeze(0)
    #         print("时间点---:", time.time())
    #
    #         print("时间点2:", time.time())
    #         # 计算当前批次中每个子集合的 Hausdorff 距离
    #         start_idx = 0
    #         for j, length in enumerate(set_lengths_batch):
    #             end_idx = start_idx + length
    #             dist_subset = dist_matrix[:, start_idx:end_idx]
    #             d_AB = torch.max(torch.min(dist_subset, dim=1)[0])
    #             d_BA = torch.max(torch.min(dist_subset, dim=0)[0])
    #             hausdorff_distances[i + j] = torch.max(d_AB, d_BA).item()
    #             start_idx = end_idx
    #         print("时间点3:", time.time())
    #
    #     return hausdorff_distances