


# def compare_benchmark(real_distances, search_result, recall_all):
#     """
#     比较基准数据集的结果。
#     """
#     for topK in recall_all.keys():
#         real = list(sorted(range(len(real_distances)), key=lambda i: real_distances[i]))
#         search = [item for item in search_result]
#
#         recall_all[topK].append(len(set(real[:topK]) & set(search[:topK])) / topK)
#
#     return recall_all



import torch
def compare_benchmark(real_distances, search_result, recall_all):
    """
    比较基准数据集的结果。
    """
    # 使用 torch.topk 找到前 k 个最小值及其索引
    topk_values, real = torch.topk(real_distances, 10, largest=False)
    search = torch.tensor(search_result)

    for topK in recall_all.keys():
        recall_all[topK].append(len(set(real[:topK].tolist()) & set(search[:topK].tolist())) / topK)


    return recall_all
