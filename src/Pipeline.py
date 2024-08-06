import time

import torch

from src import Timer, logger




class Pipeline:
    def __init__(self, filters, distInstance, dataloader):
        """初始化Pipeline"""

        # 过滤器：传入多个实例
        self.filters = filters
        # 距离计算实例：传入一个实例
        self.distInstance = distInstance
        # 数据加载器：传入一个实例
        self.dataloader = dataloader

    def query(self, query_index, topK=10):
        """查询"""
        # 第一步：过滤
        candidate_index_list = self._filter(query_index)

        # 第二步：精确计算
        with Timer("Exact Hausdorff Distance", dic_key="exact_hausdorff_time"):
            distances = self.distInstance.cal_p2b_distance(query_index, candidate_index_list,
                                                                self.dataloader)
        # 第三步：根据真实距离排序，获取最终的索引
        sorted_indices = torch.argsort(distances)[:topK]

        return [candidate_index_list[r].item() for r in sorted_indices]

    def _filter(self, query_index):
        """过滤"""

        candidate_index_list = None
        for filter_instance in self.filters:
            with Timer("Filter: {}".format(filter_instance.__class__.__name__), dic_key="{}_filter_time".format(filter_instance.__class__.__name__)):
                candidate_index_list = filter_instance.get_candidate(query_index, candidate_index_list)

            logger.info(f"{filter_instance.__class__.__name__}, candidate_num: {len(candidate_index_list)}.")

        return candidate_index_list

