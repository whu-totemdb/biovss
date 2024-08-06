import time

from src import logger, Timer
from src.DataLoader.DataLoader import DataLoader
from src.Filter.BloomGraph import BloomGraph
from src.Filter.IndexCount import IndexCount
from src.Filter.OverlapVector import OverlapVector
from src.Pipeline import Pipeline
from src.Refinement.ExactHausdorff import ExactHausdorff
from src.Refinement.ParallelExactHausdorff import ParallelExactHausdorff

from utils import compare_benchmark
import torch

def main():
    config = {
        "basePath": "/home/brucelee/code/IndexInvertedCount",
        "device": "cpu",
        "dataSet": "CS",
        "hid": 1024,
        "wta": 64,  # CS的2048的:16,32,48,64。 Medicine的2048的：16,32,48,64
        "IndexCount:minCountValue": 1, # 最小的countValue，大于等于
        "IndexCount:filter_method": "filter_decay_union",
        "IndexCount:union_index_num": 3, # 探测的倒排索引的数量,3的时候还不错（CS设置的3）, Medicine设置6

        "OverlapVector:candidate_num": 50000, # 候选集的数量

        "BloomGraph:candidate_num": 100,  # 候选集的数量,每个查询点
    }
    # log配置参数
    logger.info("倒排索引访问数量的实验")
    logger.info(config)

    # 加载基本数据
    dataLoader = DataLoader(config)
    query_index_list = dataLoader.benchmark_hausdorff["indices"]
    real_distances_list = torch.tensor(dataLoader.benchmark_hausdorff["distances"])


    # 第一步：定义过滤的实例
    indexCount = IndexCount(config, dataLoader, load_from_file=True)
    overlapVector = OverlapVector(config, dataLoader)
    # bloomGraph = BloomGraph(dataLoader, indexCount, config["BloomGraph:candidate_num"], device=config["device"])

    # 第二步：定义Pipeline
    filters = [indexCount, overlapVector]
    # filters = [bloomGraph]
    distInstance = ParallelExactHausdorff(dataLoader, device=config["device"])
    pipeline = Pipeline(filters, distInstance, dataLoader)

    # 第三步：进行搜索
    for query_index, real_distances in zip(query_index_list, real_distances_list):
        logger.info(f"=====================================开始搜索:{query_index}=================================================")
        with Timer("Total search time:", dic_key="query_time"):
            search_result = pipeline.query(query_index, topK= 10)

        # 输出准确率
        recall = {k: sum(v) / len(v) for k, v in compare_benchmark(real_distances, search_result).items()}
        logger.dict_log["mean_recall(all_query)"] = recall
        logger.info(f"topk的平均准确率(all_query)：{recall}")

        logger.dic_jsonl() # 将用于模型训练的json数据导入到文件中
        logger.dict_clear() # 清空字典日志




if __name__ == '__main__':
    main()