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

torch::Tensor cal_p2b_distance_lsh(
    int64_t query_index,
    const torch::Tensor& candidate_index_set,
    const torch::Tensor& query_set,
    const torch::Tensor& cumulative_offsets,
    const torch::Tensor& set_lengths,
    const torch::Tensor& concatenated_matrix,
    const std::string& device_str
) {
    // Ensure tensors are on CPU
    auto query_set_cpu = query_set.to(torch::kCPU);
    auto concatenated_matrix_cpu = concatenated_matrix.to(torch::kCPU);

    int64_t batch_size = 50000;
    torch::Tensor hausdorff_distances = torch::ones({candidate_index_set.size(0)}, torch::TensorOptions().device(torch::kCPU));

    // Compute minimum distance from each query point to its candidate set
    for (int64_t i = 0; i < candidate_index_set.size(0); i += batch_size) {
        auto candidate_index_batch = candidate_index_set.narrow(0, i, std::min(batch_size, candidate_index_set.size(0) - i)).to(torch::kCPU);

        auto cumulative_offsets_batch = cumulative_offsets.index_select(0, candidate_index_batch);
        auto set_lengths_batch = set_lengths.index_select(0, candidate_index_batch);

        std::vector<int64_t> vectors_index_vector;
        for (int64_t j = 0; j < candidate_index_batch.size(0); ++j) {
            int64_t start = cumulative_offsets_batch[j].item<int64_t>() - set_lengths_batch[j].item<int64_t>();
            int64_t end = cumulative_offsets_batch[j].item<int64_t>();
            for (int64_t k = start; k < end; ++k) {
                vectors_index_vector.push_back(k);
            }
        }

        auto vectors_index_batch = torch::tensor(vectors_index_vector, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        auto concatenated_matrix_batch = concatenated_matrix.index_select(0, vectors_index_batch);

        // Compute Hamming distance between query_set and candidate sets
        auto xor_result_matrix = torch::bitwise_xor(query_set_cpu.unsqueeze(1), concatenated_matrix_batch.unsqueeze(0));

        std::vector<int64_t> hamming_distances(xor_result_matrix.size(0) * xor_result_matrix.size(1));

        #pragma omp parallel for
        for (int64_t j = 0; j < xor_result_matrix.size(0); ++j) {
            for (int64_t k = 0; k < xor_result_matrix.size(1); ++k) {
                auto xor_result = xor_result_matrix[j][k];
                int64_t hamming_distance = 0;
                for (int64_t bit = 0; bit < xor_result.numel(); ++bit) {
                    hamming_distance += __builtin_popcount(xor_result[bit].item<uint8_t>());
                }
                hamming_distances[j * xor_result_matrix.size(1) + k] = hamming_distance;
            }
        }

        auto dist_matrix = torch::from_blob(hamming_distances.data(), {xor_result_matrix.size(0), xor_result_matrix.size(1)}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

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

        #pragma omp parallel for
        for (int64_t j = 0; j < set_lengths_batch.size(0); ++j) {
            auto dist_subset = dist_matrix.slice(1, start_indices[j], end_indices[j]);
            auto d_AB = std::get<0>(torch::min(dist_subset, 1)).max();
            auto d_BA = std::get<0>(torch::min(dist_subset, 0)).max();
            hausdorff_distances[i + j] = torch::max(d_AB, d_BA);
        }
    }

    return hausdorff_distances;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cal_p2b_distance_lsh", &cal_p2b_distance_lsh, "Calculate p2b distance");
}
