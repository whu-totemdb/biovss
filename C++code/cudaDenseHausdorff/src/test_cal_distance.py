import torch
import hamming_flat

def test_hamming_knn():
    # 生成示例数据
    database = torch.tensor([[0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                             [2, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                             [255, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.uint8)
    query = database[0]  # 仅查询第一个向量

    # 添加数据库
    hamming_flat.add_database(database)

    # 定义数据库向量索引
    database_indices = torch.tensor([0, 1, 2], dtype=torch.int32)

    # 设置返回的最近邻数量
    k = 3

    # 调用 C++ 扩展
    result = hamming_flat.hamming_knn(query, database_indices, k)

    # 打印结果
    print(result.shape)
    print("Hamming KNN Result:")
    for i in range(k):
        dist = result[i, 0].item()
        idx = result[i, 1].item()
        print(f"Distance: {dist}, Index: {idx}")

if __name__ == "__main__":
    test_hamming_knn()
