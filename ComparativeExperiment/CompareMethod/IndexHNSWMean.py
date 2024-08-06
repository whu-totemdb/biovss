import time

import numpy as np
import torch
import faiss

import cal_distance



class IndexHNSWMean():
    def __init__(self, dataloader = None, num_candidate = 20000, device = "cpu"):
        self.num_candidate = num_candidate
        self.device = device
        assert dataloader is not None, "dataloader is None. 需要预先加载，为后续计算做准备"

        self.concatenated_matrix = dataloader.concat_dense_vector_matrix["concatenated_matrix"]
        self.cumulative_offsets = dataloader.concat_dense_vector_matrix["cumulative_offsets"]
        self.set_lengths = dataloader.concat_dense_vector_matrix["set_lengths"]
        self.dense_author_vectors = dataloader.dense_author_vectors

        # 计算平均向量用于构建倒排索引
        self.author_vectors_dense_mean = torch.stack([torch.mean(author_vector, dim=0) for author_vector in self.dense_author_vectors])

        # 用于统计该方法平均召回率
        self.recall_all = {3: [], 5: [], 10: []}

        # 初始化IndexIVFFlat索引
        self.index_IndexHNSW = None
        self._init_index()

    def query(self, query_index, topK):
        # 第一步：通过索引来筛选候选集合
        query = self.author_vectors_dense_mean[query_index: query_index+1].numpy()
        _, search_result = self.index_IndexHNSW.search(query, self.num_candidate)

        # 第二步：根据候选点计算真实距离
        search_result = torch.from_numpy(search_result[search_result != -1].astype(np.int32))
        print("候选的数量为：", len(search_result))
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



    def _init_index(self):
        """初始化索引"""
        dim = self.author_vectors_dense_mean.size(1)
        data = self.author_vectors_dense_mean.numpy()

        # 建立索引
        self.index_IndexHNSW = faiss.IndexHNSWFlat(dim, 2048)  # 32 是 M 的值，表示每个节点的近邻数量

        # 调整 IndexHNSW 参数
        # self.index_IndexHNSW.hnsw.efConstruction = 200  # 建立索引时的 ef 参数
        self.index_IndexHNSW.hnsw.efSearch = 200  # 搜索时的 ef 参数

        # 将数据添加到索引中
        self.index_IndexHNSW.add(data)

