from ComparativeExperiment import logger, Timer
from ComparativeExperiment.utils import compare_benchmark


class CompareBase:
    def __init__(self, compare_method_instances = []):
        """初始化对比方法"""
        assert len(compare_method_instances) > 0, "对比方法为空"

        self.compare_method_instances = compare_method_instances


    def query_experiment(self, real_distances, query_index, topK=10):
        """查询"""
        for compare_method_instance in self.compare_method_instances:
            # 获取当前实例的名称
            method_name = compare_method_instance.__class__.__name__
            with Timer(f"{method_name}-Total search time:", dic_key=f"{method_name}-query_time"):
                search_result = compare_method_instance.query(query_index, topK)

            # 输出准确率
            recall = {k: sum(v) / len(v) for k, v in compare_benchmark(real_distances, search_result,
                                                                       compare_method_instance.recall_all).items()}
            logger.dict_log[f"{method_name}-mean_recall(all_query)"] = recall
            logger.info(f"{method_name}-(all_query)-各个topk的平均准确率：{recall}")



