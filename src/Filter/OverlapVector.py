import os

import torch.autograd.profiler as profiler
import torch

import hamming_flat
class OverlapVector:
    def __init__(self, config, dataLoader):
        self.config = config
        self.dataLoader = dataLoader

        hamming_flat.add_database(self.dataLoader.single_author_vectors)
        print("汉明索引数据集加载完成")

    def get_candidate(self, query_index, candidate_index_list):
        return self._method_filter(query_index, candidate_index_list)

    def _method_filter(self, query_index, candidate_index_list):
        """过滤获取topk个结果"""
        query_single_vector = self.dataLoader.single_author_vectors[query_index]

        # 计算汉明距离
        result = hamming_flat.hamming_knn(query_single_vector, candidate_index_list, self.config["OverlapVector:candidate_num"])

        return result[:, 1]

    # def _method_filter(self, query_index, candidate_index_list):
    #     """过滤获取topk个结果"""
    #     query_single_vector = self.dataLoader.single_author_vectors[query_index]
    #     author_single_vectors = self.dataLoader.single_author_vectors[candidate_index_list]
    #
    #
    #
    #     return search_result
    #
    # def _init_IndexBinaryFlat(self):
    #     """初始化IndexBinaryFlat"""
    #     # 添加向量到索引中
    #     self.index.add(self.dataLoader.single_author_vectors)
    # #        with profiler.profile(use_cuda=False) as prof:
    #         dist = torch.sum(torch.logical_xor(query_single_vector, author_single_vectors.unsqueeze(0)),
    #                          dim=2).squeeze(0)
    #         search_result = candidate_index_list[dist.argsort()[:self.config["OverlapVector:candidate_num"]].to("cpu")]
    #     print(prof.key_averages().table(sort_by="cuda_time_total"))
