import torch

from ComparativeExperiment import Timer, logger
from ComparativeExperiment.CompareBase import CompareBase
from ComparativeExperiment.CompareMethod.BruceExactHausdorff import BruceExactHausdorff
from ComparativeExperiment.CompareMethod.IndexHNSWMean import IndexHNSWMean
from ComparativeExperiment.CompareMethod.IndexIVFFlatMean import IndexIVFFlatMean
from ComparativeExperiment.CompareMethod.IndexIVFPQMean import IndexIVFPQMean
from ComparativeExperiment.CompareMethod.IndexIVFScalarQuantizerMean import IndexIVFScalarQuantizerMean
from ComparativeExperiment.DataLoader.DataLoader import DataLoader
from ComparativeExperiment.utils import compare_benchmark


from _NaiveBioVSS.main import NaiveBioVSS

def main_c():
    config = {
        "basePath": "/home/brucelee/code/IndexInvertedCount",
        "device": "cpu",
        "dataSet": "Medicine",

        "indexIVFFlatMean:n_clusters": 50,

        "indexHNSWMean:num_candidate": 200000,

        "indexIVFPQMean:n_clusters": 15,
        "indexIVFPQMean:num_candidate": 50000,
        "indexIVFPQMean:m": 16,
        "indexIVFPQMean:nbits": 8,

        "indexIVFScalarQuantizerMean:num_bits": 32,
        "indexIVFScalarQuantizerMean:nlist": 15,
        "indexIVFScalarQuantizerMean:nprobe": 2,
        "indexIVFScalarQuantizerMean:num_candidate": 1000,
    }
    logger.info(f"config: {config}")

    # 加载基本数据
    dataLoader = DataLoader(config)
    query_index_list = dataLoader.benchmark_hausdorff["indices"]
    real_distances_list = torch.tensor(dataLoader.benchmark_hausdorff["distances"])

    # 第一步：初始化对比方法
    # bruceExactHausdorff = BruceExactHausdorff(dataLoader, config["device"])
    # indexIVFFlatMean = IndexIVFFlatMean(dataLoader, config["indexIVFFlatMean:n_clusters"], num_candidate = 200000, device = config["device"])
    # ======num_candidate = 200000是为了将IVF筛选出来的所有数据点都考虑到，只控制聚类大小来筛选数据

    # indexHNSWMean = IndexHNSWMean(dataLoader, config["indexHNSWMean:num_candidate"], device = config["device"])
    # indexIVFPQMean = IndexIVFPQMean(
    #     dataLoader,
    #     n_clusters=config["indexIVFPQMean:n_clusters"],
    #     num_candidate=config["indexIVFPQMean:num_candidate"],
    #     m=config["indexIVFPQMean:m"],
    #     nbits=config["indexIVFPQMean:nbits"],
    #     device=config["device"]
    # )
    #


    # indexIVFScalarQuantizerMean = IndexIVFScalarQuantizerMean(
    #     dataloader=dataLoader,
    #     num_bits=config["indexIVFScalarQuantizerMean:num_bits"],
    #     num_candidate=config["indexIVFScalarQuantizerMean:num_candidate"],
    #     nlist=config["indexIVFScalarQuantizerMean:nlist"],
    #     nprobe=config["indexIVFScalarQuantizerMean:nprobe"],
    #     device=config["device"]
    # )

    naiveBioVSS = NaiveBioVSS(config, dataLoader)

    # 第二步：打包对比方法
    compare_method_instances = [naiveBioVSS]
    compareBase = CompareBase(compare_method_instances)

    # 第三步：进行搜索
    for query_index, real_distances in zip(query_index_list, real_distances_list):
        logger.info(f"=====================================开始搜索:{query_index}=================================================")
        compareBase.query_experiment(real_distances, query_index, topK= 10) # 除了naive的方法是top30，其它都是top10

        logger.dic_jsonl() # 将用于模型训练的json数据导入到文件中
        logger.dict_clear() # 清空字典日志

if __name__ == '__main__':
    main_c()
