#include <torch/torch.h>
#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <omp.h>


void print_current_time(const std::string& label) {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::tm local_tm = *std::localtime(&now_c);
    std::cout << label << ": "
              << std::put_time(&local_tm, "%H:%M:%S")
              << '.' << std::setw(3) << std::setfill('0') << now_ms.count()
              << std::endl;
}





torch::Tensor cal_p2b_distance(
    int64_t query_index,
    const torch::Tensor& candidate_index_set,
    const torch::Tensor& query_set,
    const torch::Tensor& cumulative_offsets,
    const torch::Tensor& set_lengths,
    const torch::Tensor& concatenated_matrix,
    const std::string& device_str
) {
    torch::Device device(device_str);
    auto query_set_device = query_set.to(device);

    int64_t batch_size = 50000;
    torch::Tensor hausdorff_distances = torch::ones({candidate_index_set.size(0)}, torch::TensorOptions().device(device));

    // 计算每个查询点到其候选集的最小距离
    for (int64_t i = 0; i < candidate_index_set.size(0); i += batch_size) {
//        print_current_time("时间点0---");

        // 获取当前批次的候选作者索引
        auto candidate_index_batch = candidate_index_set.narrow(0, i, std::min(batch_size, candidate_index_set.size(0) - i)).to(device);

//        print_current_time("时间点1---");

        // 使用 index_select 方法选择特定索引的子张量
        auto cumulative_offsets_batch = cumulative_offsets.index_select(0, candidate_index_batch);
        auto set_lengths_batch = set_lengths.index_select(0, candidate_index_batch);

//        print_current_time("时间点2---");

        // 获取当前批次的候选向量索引和对应的向量矩阵
        std::vector<int64_t> vectors_index_vector;
        for (int64_t j = 0; j < candidate_index_batch.size(0); ++j) {
            int64_t start = cumulative_offsets_batch[j].item<int64_t>() - set_lengths_batch[j].item<int64_t>();
            int64_t end = cumulative_offsets_batch[j].item<int64_t>();
            for (int64_t k = start; k < end; ++k) {
                vectors_index_vector.push_back(k);
            }
        }

//        print_current_time("时间点3---");

        auto vectors_index_batch = torch::tensor(vectors_index_vector, torch::TensorOptions().dtype(torch::kInt64).device(device));
        auto concatenated_matrix_batch = concatenated_matrix.index_select(0, vectors_index_batch);


//        print_current_time("时间点---");

        // 计算查询集合与候选集合之间的距离矩阵
        auto dist_matrix = torch::cdist(query_set_device.unsqueeze(0), concatenated_matrix_batch.unsqueeze(0)).squeeze(0);
//        print_current_time("时间点---");


        // 计算当前批次中每个子集合的 Hausdorff 距离: start_idx要在for中进行分离才能并行计算
//        print_current_time("时间点4");
        // 预先计算所有的 start_idx 和 end_idx
        std::vector<int64_t> start_indices(set_lengths_batch.size(0));
        std::vector<int64_t> end_indices(set_lengths_batch.size(0));
        int64_t start_idx = 0;
        for (int64_t j = 0; j < set_lengths_batch.size(0); ++j) {
            int64_t length = set_lengths_batch[j].item<int64_t>();
            int64_t end_idx = start_idx + length;
            start_indices[j] = start_idx;
            end_indices[j] = end_idx;
            start_idx = end_idx;
        }

//        print_current_time("时间点5");
        // 使用 OpenMP 并行化处理 Hausdorff 距离计算
        #pragma omp parallel for
        for (int64_t j = 0; j < set_lengths_batch.size(0); ++j) {
            auto dist_subset = dist_matrix.slice(1, start_indices[j], end_indices[j]);
            auto d_AB = std::get<0>(torch::min(dist_subset, 1)).max();
            auto d_BA = std::get<0>(torch::min(dist_subset, 0)).max();
            hausdorff_distances[i + j] = torch::max(d_AB, d_BA);
        }
//        print_current_time("时间点6");
    }

    return hausdorff_distances;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cal_p2b_distance", &cal_p2b_distance, "Calculate p2b distance");
}
