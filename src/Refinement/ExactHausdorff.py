import torch

from scipy.spatial.distance import directed_hausdorff

class ExactHausdorff():
    def __init__(self, device = "cpu"):
        pass

    def cal_p2p_distance(self):
        """计算单项之间的距离"""
        pass

    def cal_p2b_distance(self, query_index, candidate_index_set, dataloader):
        """计算单项对多项之间的距离"""
        query_set = dataloader.dense_author_vectors[query_index]

        # 初始化一个用于存储每个查询点到其候选集最小距离的数组
        distances = torch.ones(len(candidate_index_set))

        # 计算每个查询点到其候选集的最小距离
        for i, index in enumerate(candidate_index_set):
            candidate_set = dataloader.dense_author_vectors[index]
            d1 = directed_hausdorff(query_set, candidate_set)[0]
            d2 = directed_hausdorff(candidate_set, query_set)[0]
            distances[i] = max(d1, d2)

        return distances

    def cal_b2b_distance(self):
        """计算多项对多项的距离矩阵"""
        pass


