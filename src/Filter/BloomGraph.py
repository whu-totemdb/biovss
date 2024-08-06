import concurrent.futures
import numpy as np
import torch
import faiss
import resource

class BloomGraph:
    def __init__(self, dataloader=None, indexCount=None, num_candidate=20000, device="cpu"):
        self.num_candidate = num_candidate
        self.device = device
        self.indexCount = indexCount  # 用于获取倒排索引
        self.dataLoader = dataloader
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
        self.index_IndexHNSW_list = []
        self.index_Raw_list = []
        self._init_index()

    def get_candidate(self, query_index, graph_index_list):
        return self._method_filter(query_index)

    def _method_filter(self, query_index):
        """过滤获取topk个结果"""
        query = self.dataLoader.single_author_vectors[query_index:query_index + 1].numpy()
        search_results_all = []
        for graph_index in self._get_graph_index_list(query_index):
            # 对当前图进行搜索
            _, search_result = self.index_IndexHNSW_list[graph_index].search(query, self.num_candidate)
            search_result = torch.from_numpy(search_result[search_result != -1])

            # 还原为原始的索引
            search_result = self.index_Raw_list[graph_index][search_result].astype(np.int32)
            search_results_all.append(search_result)
        # 将结果合并为一个 numpy 数组
        search_results_all = np.vstack(search_results_all)

        return search_results_all

    def _get_graph_index_list(self, query_index):
        query_count_vector = self.indexCount.count_author_vectors[query_index]

        # 对query访问的索引顺序进行排序：降序排列
        query_sorted_count, query_indices = torch.sort(query_count_vector, descending=True)

        return query_indices[:3]

    def _init_single_index(self, i, single_author_vectors):
        # 从倒排索引中获取对应的索引
        end_index = self.indexCount.indexIVCount_minCountValue[i].item()
        index_current = self.indexCount.indexIVCount[i][:end_index]

        # 根据索引获取当前的数据
        data = single_author_vectors[index_current].float()
        dim = data.size(1)

        # 建立索引
        index_IndexHNSW_i = faiss.IndexHNSWFlat(dim, 32)
        # index_IndexHNSW_i.hnsw.efConstruction = 200
        # index_IndexHNSW_i.hnsw.efSearch = 200

        # 将数据添加到索引中
        index_IndexHNSW_i.add(data.numpy())

        return index_current, index_IndexHNSW_i

    def _init_index(self):
        """初始化索引"""

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))  # 或者更高的值

        single_author_vectors = self.dataLoader.single_author_vectors  # 在主进程中加载数据

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(self._init_single_index, i, single_author_vectors)
                for i in range(len(self.indexCount.indexIVCount))
            ]

            for future in concurrent.futures.as_completed(futures):
                index_current, index_IndexHNSW_i = future.result()
                self.index_Raw_list.append(index_current)
                self.index_IndexHNSW_list.append(index_IndexHNSW_i)
                print("IndexHNSW_", self.index_Raw_list.index(index_current), "初始化完成")
