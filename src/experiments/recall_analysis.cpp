// eval_main.cpp
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <CLI/CLI.hpp>
#include <omp.h>

#include "index/nmf_index.h"
#include "index/backend/naive.h"
#include "index/backend/adaptive.h" // Include your new backend
#include "utils/hdf5_loader.h"
#include "utils/index_eval.h"       // (Matches your nmf_evaluation.h)

int main(int argc, char** argv) {
    CLI::App app{"NMF Index Evaluator"};

    std::string index_path, data_path;
    std::string backend_type = "adaptive"; // Default to your new fast backend
    int eval_k = 100, threads = 8;

    app.add_option("-i,--index", index_path, "Path to the saved NMF index (.h5)")->required();
    app.add_option("-d,--data", data_path, "Path to dataset (.h5)")->required();
    app.add_option("-b,--backend", backend_type, "Backend to use: 'naive' or 'adaptive'");
    app.add_option("-k,--top-k", eval_k, "Top K results to evaluate");
    app.add_option("-t,--threads", threads, "Number of threads to use for search");

    CLI11_PARSE(app, argc, argv);

    try {
        omp_set_num_threads(threads);

        std::cout << "[Eval] Loading index from " << index_path << "...\n";

        std::unique_ptr<NMFIndex> index;
        if (backend_type == "naive") {
            auto factory = [](const Eigen::MatrixXf& H, const Eigen::MatrixXi& lists) {
                return std::make_unique<NaiveIVFBackend>(H, lists);
            };
            index = NMFIndex::load(index_path, factory);
        } else if (backend_type == "adaptive") {
            auto factory = [](const Eigen::MatrixXf& H, const Eigen::MatrixXi& lists) {
                return std::make_unique<AdaptiveIVFBackend>(H, lists);
            };
            index = NMFIndex::load(index_path, factory);
        } else {
            throw std::invalid_argument("Unknown backend: " + backend_type);
        }

        std::cout << "[Eval] Loading data from " << data_path << "...\n";
        HDF5Loader loader(data_path);
        auto queries = loader.loadSparse<float>("otest/queries");
        auto gt      = loader.loadGroundTruth("otest/knns");
        auto X_docs  = loader.loadSparse<float>("train");

        // Dynamically build parameter configurations for the evaluation and sweep
        std::unique_ptr<IVFBackend::SearchParams> default_params;
        std::string default_config_name;
        std::vector<std::pair<std::string, std::unique_ptr<IVFBackend::SearchParams>>> sweep_configs;

        if (backend_type == "naive") {
            default_config_name = "nprobe=16";
            default_params = std::make_unique<NaiveIVFBackend::SearchParams>(10, 0.5f);

            for (int np : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50}) {
                std::string name = "np=" + std::to_string(np);
                sweep_configs.push_back({name, std::make_unique<NaiveIVFBackend::SearchParams>(np, 0.5f)});
            }
        } else {
            // Adaptive configuration
            default_config_name = "miss=30, dr=0.15";
            default_params = std::make_unique<AdaptiveIVFBackend::SearchParams>(30, 0.15f);

            // Sweep combinations of (max_misses, drop_ratio)
            // Sweep combinations of (max_misses, drop_ratio)
            std::vector<std::pair<int, float>> adapt_sweep = {
                // Focus on speed (can we keep R@30 > 90% while dropping under 0.25ms?)
                {60, 0.30f},
                {80, 0.30f},

                // The current champion baseline
                {80, 0.20f},

                // Focus on Recall (can we hit 95% R@30 while keeping under 0.5ms?)
                {100, 0.20f},
                {120, 0.15f},
                {100, 0.10f}
            };
            for (auto [m, dr] : adapt_sweep) {
                char buffer[32];
                snprintf(buffer, sizeof(buffer), "m=%d, dr=%.2f", m, dr);
                sweep_configs.push_back({buffer, std::make_unique<AdaptiveIVFBackend::SearchParams>(m, dr)});
            }
        }

        // Convert to raw pointers for the generic evaluator
        std::vector<std::pair<std::string, const IVFBackend::SearchParams*>> sweep_ptrs;
        for (const auto& conf : sweep_configs) {
            sweep_ptrs.push_back({conf.first, conf.second.get()});
        }

        evaluate_nmf_index(*index, queries, X_docs, gt, {10, 30, 100}, eval_k,
                           default_config_name, default_params.get(), sweep_ptrs);

    } catch(const std::exception& e) {
        std::cerr << "\n[Fatal Error] " << e.what() << "\n";
        return 1;
    }
    return 0;
}