import os.path
from itertools import accumulate

import numpy as np
import torch


class DataLoader:
    def __init__(self, config):
        self.config = config

        self.count_author_vectors = self.load_count_author_vectors()
        self.dense_author_vectors = self.load_dense_author_vectors()
        self.concat_dense_vector_matrix = self.load_concat_dense_vector_matrix()
        # self.binary_author_vectors = self.load_binary_author_vectors()
        self.single_author_vectors = self.load_single_author_vectors()

        # 加载benchmark
        self.benchmark_hausdorff = self.load_benchmark_hausdorff()

    def load_count_author_vectors(self):
        file_name = f"{self.config.get('dataSet')}_hid_{self.config.get('hid')}_wta_{self.config.get('wta')}_count_author_vectors.pkl"
        print(file_name)
        path_count_author_vectors = os.path.join(self.config.get("basePath"), "data", self.config.get("dataSet"),
                                                 file_name)

        return torch.load(path_count_author_vectors)

    def load_dense_author_vectors(self):
        with open(f'/data1/brucelee/AuthorBinaryData/{self.config.get("dataSet")}/author_real_normalization_vectors.pickle', 'rb') as file:
            author_vectors_dense = torch.load(file)
            print("作者的稠密向量加载完成，作者向量集大小为:", len(author_vectors_dense))
        return author_vectors_dense

    def load_binary_author_vectors(self):
        return None

    def load_single_author_vectors(self):
        file_name = f"{self.config.get('dataSet')}_hid_{self.config.get('hid')}_wta_{self.config.get('wta')}_single_author_vectors.pkl"
        path_single_author_vectors = os.path.join(self.config.get("basePath"), "data/dataSet", self.config.get("dataSet"),
                                                 file_name)

        # 读取后，将其转换为uint8的二进制数据
        single_author_vectors = torch.load(path_single_author_vectors).numpy()
        single_author_vectors = np.packbits(single_author_vectors, axis=1)
        single_author_vectors = torch.tensor(single_author_vectors, dtype=torch.uint8)
        return single_author_vectors.to(self.config["device"])

    def load_benchmark_hausdorff(self):
        """加载基准数据"""
        size = 500
        benchmarkPath = os.path.join(self.config['basePath'], f"data/benchmark/{self.config['dataSet']}",
                                     f"{self.config['dataSet']}_hausdorff_{size}.pickle")
        return torch.load(benchmarkPath)

    def load_concat_dense_vector_matrix(self):
        concatenated_matrix  = torch.cat(self.dense_author_vectors, dim=0).to(self.config["device"])
        set_lengths = torch.tensor([setB.shape[0] for setB in self.dense_author_vectors], dtype=torch.int32).to(self.config["device"])
        cumulative_offsets = torch.tensor(list(accumulate(set_lengths)), dtype=torch.int32).to(self.config["device"])
        print("load_concat_dense_vector_matrix 完成")
        return {"concatenated_matrix": concatenated_matrix , "cumulative_offsets": cumulative_offsets, "set_lengths": set_lengths}
