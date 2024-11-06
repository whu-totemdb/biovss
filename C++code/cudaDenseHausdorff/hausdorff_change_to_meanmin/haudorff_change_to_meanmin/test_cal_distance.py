# -*- coding: utf-8 -*-

import torch
import cal_distance_meanmin

# 示例数据


query_tensor = torch.rand(1, 50).cpu()  # 查询张量
candidate_index_set = torch.arange(200).cpu()  # 候选索引
cumulative_offsets = torch.arange(1, 201).cpu()
set_lengths = torch.randint(1, 2, (200,)).cpu()
concatenated_matrix = torch.rand(200, 50).cpu()

# 调用C++函数进行计算
distances = cal_distance_meanmin.cal_p2b_distance(
    0,
    candidate_index_set,
    query_tensor,
    cumulative_offsets,
    set_lengths,
    concatenated_matrix,
    'cpu'
)

print(distances)
