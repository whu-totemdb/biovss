#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <omp.h>

static torch::Tensor database;  // 用于存储数据库向量

// 添加数据库向量
void add_database(const torch::Tensor& db) {
    database = db;
}

// 计算两个二进制向量之间的汉明距离
int32_t hamming_distance(const uint8_t* a, const uint8_t* b, size_t length) {
    int32_t dist = 0;
    for (size_t i = 0; i < length; ++i) {
        dist += __builtin_popcount(a[i] ^ b[i]);
    }
    return dist;
}

// 高效汉明距离计算
torch::Tensor hamming_knn(
        const torch::Tensor& query,
        const torch::Tensor& database_indices,
        int64_t k) {

    auto query_ptr = query.data_ptr<uint8_t>();
    auto database_ptr = database.data_ptr<uint8_t>();
    auto database_indices_ptr = database_indices.data_ptr<int32_t>();

    size_t bytes_per_code = database.size(1);
    size_t num_database = database_indices.size(0);

    // 创建一个用于存储结果的张量，2列：一列存储距离，一列存储索引
    auto options = torch::TensorOptions().dtype(torch::kInt32);
    k = std::min(k, (int64_t)num_database);
    torch::Tensor result = torch::empty({k, 2}, options);
    auto result_ptr = result.data_ptr<int32_t>();

    // 线程局部存储每个线程的最小堆
    std::vector<std::vector<std::pair<int32_t, int64_t>>> local_heaps(omp_get_max_threads());

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        auto& heap = local_heaps[thread_id];
        heap.reserve(k);

        #pragma omp for nowait
        for (int64_t idx = 0; idx < num_database; ++idx) {
            int64_t j = database_indices_ptr[idx];
            const uint8_t* db_ptr = database_ptr + j * bytes_per_code;
            int32_t dist = hamming_distance(query_ptr, db_ptr, bytes_per_code);

            if (heap.size() < k) {
                heap.emplace_back(dist, j);
                std::push_heap(heap.begin(), heap.end(), std::less<>());
            } else if (dist < heap.front().first) {
                std::pop_heap(heap.begin(), heap.end(), std::less<>());
                heap.back() = std::make_pair(dist, j);
                std::push_heap(heap.begin(), heap.end(), std::less<>());
            }
        }
    }

    // 合并所有线程的最小堆
    std::vector<std::pair<int32_t, int64_t>> global_heap;
    global_heap.reserve(k);

    for (const auto& heap : local_heaps) {
        for (const auto& elem : heap) {
            if (global_heap.size() < k) {
                global_heap.push_back(elem);
                std::push_heap(global_heap.begin(), global_heap.end(), std::less<>());
            } else if (elem.first < global_heap.front().first) {
                std::pop_heap(global_heap.begin(), global_heap.end(), std::less<>());
                global_heap.back() = elem;
                std::push_heap(global_heap.begin(), global_heap.end(), std::less<>());
            }
        }
    }

    std::sort_heap(global_heap.begin(), global_heap.end(), std::less<>());

    // 将结果存储到返回的张量中
    for (int i = 0; i < k; ++i) {
        result_ptr[i * 2] = global_heap[i].first;       // 距离
        result_ptr[i * 2 + 1] = global_heap[i].second;  // 索引
    }

    return result;
}

// Pybind11绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_database", &add_database, "Add Database");
    m.def("hamming_knn", &hamming_knn, "Hamming KNN");
}
