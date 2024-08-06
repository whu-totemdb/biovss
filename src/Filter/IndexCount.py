import os
import time

import torch


# 性能分析必要的包
import cProfile
import pstats
from io import StringIO
# 性能分析必要的包


class IndexCount:
    def __init__(self, config, dataLoader, load_from_file=True):
        self.config = config
        self.dataLoader = dataLoader
        self.load_from_file = load_from_file

        self.count_author_vectors = self.dataLoader.count_author_vectors
        print("count_author_vectors:", self.count_author_vectors[-1])

        # 索引的路劲
        index_name = f"{self.config.get('dataSet')}_hid_{self.config.get('hid')}_wta_{self.config.get('wta')}_indexIVCount.pkl"
        self.path_indexIVCount = os.path.join(self.config['basePath'], "data", "index_file", "index_count", index_name)

        # 索引的相关配置
        self.indexIVCount = None # 形状为(hid, author_num)
        self.indexIVCount_minCountValue = None
        self._sorted_count = None
        self._init_index()

        # 过滤方法
        _method_map = {
            "filter_decay_union": self._filter_decay_union
        }
        self._method_filter = _method_map[self.config.get("IndexCount:filter_method")]

    def get_candidate(self, query_index, candidate_index_list):
        return self._filter_decay_union(query_index)

    def _filter_decay_union(self, query_index):
        query_count_vector = self.count_author_vectors[query_index]

        # 对query访问的索引顺序进行排序：降序排列
        query_sorted_count, query_indices = torch.sort(query_count_vector, descending=True)

        # 初始化结果张量
        res = torch.tensor([], dtype=torch.int32)

        for i in range(self.config.get("IndexCount:union_index_num")):
            # 获取访问的倒排索引
            index = query_indices[i]

            # 从倒排索引中获取对应的索引
            end_index = self.indexIVCount_minCountValue[index].item()
            index_vector = self.indexIVCount[index][:end_index]

            # 将索引添加到结果张量中
            res = torch.cat((res, index_vector), dim=0)

        # 使用torch.unique来去重
        res = torch.unique(res).to(torch.int32)

        return res

    # def get_candidate(self, query_index, candidate_index_list):
    #     return torch.tensor(list(self._method_filter(query_index)), dtype=torch.int32)
    #
    # def _filter_decay_union(self, query_index):
    #
    #     query_count_vector = self.count_author_vectors[query_index]
    #
    #     # 对query访问的索引顺序进行排序：降序排列
    #     query_sorted_count, query_indices = torch.sort(query_count_vector, descending=True)
    #
    #     res = set()
    #     for i in range(self.config.get("IndexCount:union_index_num")):
    #         # 获取访问的倒排索引
    #         index = query_indices[i]
    #
    #         # 从倒排索引中获取对应的索引
    #         end_index = self.indexIVCount_minCountValue[index].item() #// (i + 1)  # ((i + 1)**(i))
    #         index_vector = self.indexIVCount[index][:end_index].tolist()
    #
    #         # 将索引添加到res中
    #         res.update(index_vector)
    #
    #     return res

    def _init_index(self,):
        """初始化IndexCount"""
        if self.load_from_file and os.path.exists(self.path_indexIVCount):
            print("Load IndexCount from file.")
            self._load_index()

        else:
            self._build_index()
            self._save_index()

    def _build_index(self):
        """设置IndexCount"""
        # 第一步：针对所有的count位置进行排序
        sorted_count, indices = torch.sort(self.dataLoader.count_author_vectors, dim=0, descending=True)

        # 初始化count索引
        self.indexIVCount = torch.transpose(indices, 0, 1).to(torch.int32)

        # 初始化排序后的count值
        self._sorted_count = torch.transpose(sorted_count, 0, 1) # 变换后形状为(hid, author_num), 保存的是sort以后的值

        # 设置最小的count值
        self._set_minCountValue()

    def _set_minCountValue(self):
        """设置最小的count值"""
        self.indexIVCount_minCountValue = (self._sorted_count >= self.config.get("IndexCount:minCountValue")).sum(dim=1)


    def _save_index(self):
        """保存索引"""
        torch.save({
            "indexIVCount": self.indexIVCount,
            "_sorted_count": self._sorted_count
        }, self.path_indexIVCount)
        print(f"IndexCount: Index has been saved to {self.path_indexIVCount}.")

    def _load_index(self):
        """加载索引"""
        index = torch.load(self.path_indexIVCount)
        self.indexIVCount = index["indexIVCount"]
        self._sorted_count = index["_sorted_count"]
        print(f"IndexCount: Index has been loaded from {self.path_indexIVCount}.")

        # 设置最小的count值
        self._set_minCountValue()


