import torch
import cal_distance_lsh  # 替换为你实际的模块名称

# 设置较大的测试数据
query_index = 0
query_tensor = torch.randint(0, 256, (1, 50), dtype=torch.uint8).cpu()  # 查询张量，范围是0到255
candidate_index_set = torch.arange(200).cpu()  # 候选索引
cumulative_offsets = torch.arange(1, 201).cpu()
set_lengths = torch.randint(1, 2, (200,), dtype=torch.int64).cpu()
concatenated_matrix = torch.randint(0, 256, (200, 50), dtype=torch.uint8).cpu()  # 使用随机的8位打包向量，范围是0到255

# 打印输入数据
print("Query Tensor:", query_tensor)
print("Candidate Index Set:", candidate_index_set)
print("Cumulative Offsets:", cumulative_offsets)
print("Set Lengths:", set_lengths)
print("Concatenated Matrix:", concatenated_matrix)

# 调用cal_p2b_distance_lsh函数
distances = cal_distance_lsh.cal_p2b_distance_lsh(
    query_index,
    candidate_index_set,
    query_tensor,
    cumulative_offsets,
    set_lengths,
    concatenated_matrix,
    'cpu'  # 使用CPU
)

# 打印结果
print("Hausdorff Distances:", distances)
