#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <limits>

namespace py = pybind11;
// 这些是必要的头文件包含。pybind11用于创建Python扩展模块，
// 其他的是C++标准库，提供了vector、算法、整数类型和数值限制等功能。

// 创建命名空间别名，使得后面使用pybind11更方便


// 计算两个二进制向量之间的Hamming距离
uint32_t hamming_distance(const uint64_t* a, const uint64_t* b, size_t n_words) {
    uint32_t distance = 0;
    for (size_t i = 0; i < n_words; ++i) {
        distance += __builtin_popcountll(a[i] ^ b[i]);
    }
    return distance;
}

// 这个函数计算两个二进制向量之间的Hamming距离
// 参数：
// - a, b: 指向两个二进制向量的指针，每个向量被表示为一系列64位整数
// - n_words: 每个向量包含的64位整数的数量
// 实现：
// 1. 使用异或操作(^)比较两个向量的每一位
// 2. __builtin_popcountll是GCC的内建函数，用于计算64位整数中1的个数
// 3. 累加所有不同位的数量，得到Hamming距离


// 计算两个二进制集合之间的Hausdorff距离
uint32_t binary_hausdorff_distance(const uint64_t* setA, size_t setA_size,
                                   const uint64_t* setB, size_t setB_size,
                                   size_t n_words) {
    uint32_t d_AB = 0, d_BA = 0;

    // 计算从集合A到集合B的最大最小距离
    for (size_t i = 0; i < setA_size; ++i) {
        uint32_t min_dist = std::numeric_limits<uint32_t>::max();
        for (size_t j = 0; j < setB_size; ++j) {
            uint32_t dist = hamming_distance(setA + i * n_words, setB + j * n_words, n_words);
            min_dist = std::min(min_dist, dist);
        }
        d_AB = std::max(d_AB, min_dist);
    }

    // 计算从集合B到集合A的最大最小距离
    for (size_t j = 0; j < setB_size; ++j) {
        uint32_t min_dist = std::numeric_limits<uint32_t>::max();
        for (size_t i = 0; i < setA_size; ++i) {
            uint32_t dist = hamming_distance(setB + j * n_words, setA + i * n_words, n_words);
            min_dist = std::min(min_dist, dist);
        }
        d_BA = std::max(d_BA, min_dist);
    }

    return std::max(d_AB, d_BA);
}

// 这个函数计算两个二进制集合之间的Hausdorff距离
// 参数：
// - setA, setB: 指向两个二进制集合的指针
// - setA_size, setB_size: 两个集合中元素的数量
// - n_words: 每个二进制向量包含的64位整数的数量
// 实现：
// 1. 计算从A到B的距离(d_AB)：
//    - 对A中的每个元素，找到它到B中所有元素的最小距离
//    - 在这些最小距离中取最大值
// 2. 计算从B到A的距离(d_BA)，方法类似
// 3. Hausdorff距离是d_AB和d_BA的最大值



// Python接口函数
py::array_t<float> compute_hausdorff_distances(py::array_t<uint64_t> all_vectors,
                                               py::array_t<size_t> lengths,
                                               py::array_t<size_t> starts,
                                               size_t query_index,
                                               size_t vector_dim) {
    // 1. 获取输入数组的信息
    py::buffer_info buf_all_vectors = all_vectors.request();
    py::buffer_info buf_lengths = lengths.request();
    py::buffer_info buf_starts = starts.request();

    // 2. 检查输入数组的维度
    if (buf_all_vectors.ndim != 1 || buf_lengths.ndim != 1 || buf_starts.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");

    // 3. 计算一些必要的值
    size_t n_sets = buf_lengths.shape[0];
    size_t n_words = (vector_dim + 63) / 64;

    // 4. 初始化结果数组和指针
    std::vector<float> hausdorff_distances(n_sets);
    const uint64_t* ptr_all_vectors = static_cast<const uint64_t*>(buf_all_vectors.ptr);
    const size_t* ptr_lengths = static_cast<const size_t*>(buf_lengths.ptr);
    const size_t* ptr_starts = static_cast<const size_t*>(buf_starts.ptr);

    // 5. 获取查询集合的起始位置和大小
    size_t query_start = ptr_starts[query_index];
    size_t query_size = ptr_lengths[query_index];
    const uint64_t* query_vector_start = ptr_all_vectors + query_start * n_words;

    // 6. 计算所有其他集合的Hausdorff距离
    for (size_t i = 0; i < n_sets; ++i) {
        size_t current_start = ptr_starts[i];
        hausdorff_distances[i] = binary_hausdorff_distance(
            query_vector_start, query_size,
            ptr_all_vectors + current_start * n_words, ptr_lengths[i],
            n_words);
    }

    // 7. 返回结果
    return py::array_t<float>(hausdorff_distances.size(), hausdorff_distances.data());
}



// 定义Python模块
PYBIND11_MODULE(hausdorff_distance_naive_lsh, m) {
    m.doc() = "Compute Hausdorff distances for binary vectors using C++";
    m.def("compute_hausdorff_distances", &compute_hausdorff_distances,
          "Compute Hausdorff distances between a query set and all other sets");
}

// 这部分使用pybind11来定义Python模块：
// 1. PYBIND11_MODULE 宏定义了模块名称（hausdorff_distance_naive_lsh）
// 2. m.doc() 设置模块的文档字符串
// 3. m.def() 将C++函数 compute_hausdorff_distances 暴露给Python，
//    并提供一个简短的描述
