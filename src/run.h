#pragma once

#include <iostream>
#include <fstream>
#include <omp.h>
#include <Eigen/Sparse>
#include <cblas.h>
#include <optional>
#include <random>
#include <stdexcept>

#include "utils/hdf5_loader.h"

#include "index/nmf_index.h"
#include "index/backend/naive.h"
#include "index/nmf/hals_nmf.h"
#include "index/nmf/mu_nmf.h"
#include "utils/index_eval.h"

struct RunConfig {
    // Dataset
    std::string dataset   = "nq";
    std::string data_file = "";
    int         threads   = 8;

    // NMF
    std::string nmf_type      = "mu";
    int         n_components  = 3000;
    double      tol           = 1e-4;
    int         max_iter      = 30;
    double      forget_factor = 0.7;
    std::string init_method   = "acol";   // "acol" | "random"
    int         acol_p        = 5;
    bool        debug         = false;
    std::optional<int> random_state = std::nullopt;

    // Only for HALS
    int w_sweeps = 1;
    int h_sweeps = 1;

    // Index & Backend
    std::string backend_type = "naive";
    int m           = 5000;
    int nprobe      = 20;
    int sample_size = 150000;

    // File I/O
    std::string save_index_path;
    std::string save_results_path;

    // Eval
    bool evaluate_recall = false;
    std::vector<int> recall_at = {10, 30, 100};
    int              eval_k    = 100;
};

// ── Helpers ──────────────────────────────────────────────────────────────────

inline NMFBase::Init parseInitMethod(const std::string& s) {
    if (s == "acol")   return NMFBase::Init::Acol;
    if (s == "random") return NMFBase::Init::Random;
    throw std::invalid_argument("Unknown init method: '" + s + "'. Valid: acol, random");
}

inline RunConfig preset(const std::string& name) {
    RunConfig cfg;
    cfg.dataset = name;

    if (name == "nq") {
        cfg.data_file    = "data/nq.h5";
        cfg.n_components = 3000;
        cfg.max_iter     = 25;
        cfg.m            = 5000;
        cfg.nprobe       = 20;
    } else if (name == "fiqa-dev") {
        cfg.data_file    = "data/fiqa-dev.h5";
        cfg.n_components = 512;
        cfg.max_iter     = 50;
        cfg.m            = 500;
        cfg.nprobe       = 16;
    } else {
        throw std::invalid_argument("Unknown dataset preset: " + name);
    }
    return cfg;
}

// ── Execution Logic ───────────────────────────────────────────────────────────

inline int run(const RunConfig& cfg) {
    try {
        // 1. Environment Setup
        openblas_set_num_threads(cfg.threads);
        omp_set_num_threads(cfg.threads);

        const unsigned int seed = cfg.random_state.has_value()
            ? static_cast<unsigned int>(*cfg.random_state)
            : std::random_device{}();

        std::cout << "========================================\n"
                  << "Dataset       : " << cfg.dataset      << "\n"
                  << "Data File     : " << cfg.data_file    << "\n"
                  << "NMF Solver    : " << cfg.nmf_type     << "\n"
                  << "Components (k): " << cfg.n_components << "\n"
                  << "Initialization: " << cfg.init_method  << " (p=" << cfg.acol_p << ")\n"
                  << "Backend Type  : " << cfg.backend_type << "\n"
                  << "m / nprobe    : " << cfg.m << " / " << cfg.nprobe << "\n"
                  << "Threads       : " << cfg.threads << " (OMP Max: " << omp_get_max_threads() << ")\n"
                  << "Seed          : " << seed             << "\n"
                  << "========================================\n";

        // 2. Load Data
        std::cout << "[IO] Loading datasets from " << cfg.data_file << "...\n";
        HDF5Loader loader(cfg.data_file);

        auto query = loader.loadSparse<float>("otest/queries");
        auto gt    = loader.loadGroundTruth("otest/knns");
        auto train = loader.loadSparse<float>("train");

        std::cout << "[IO] Train matrix: " << train.rows() << " x " << train.cols()
                  << " (" << train.nonZeros() << " non-zeros)\n";

        // 3. Configure NMF Solver
        const NMFBase::Init init_enum = parseInitMethod(cfg.init_method);
        std::unique_ptr<NMFBase> nmf_solver;

        if (cfg.nmf_type == "hals") {
            HalsNMF::Config hals_cfg;
            hals_cfg.n_components  = cfg.n_components;
            hals_cfg.max_iter      = cfg.max_iter;
            hals_cfg.tol           = cfg.tol;
            hals_cfg.verbose       = cfg.debug;
            hals_cfg.random_state  = seed;
            hals_cfg.init_method   = init_enum;
            hals_cfg.acol_p        = cfg.acol_p;
            hals_cfg.compute_error = cfg.debug;
            hals_cfg.w_sweeps      = cfg.w_sweeps;
            hals_cfg.h_sweeps      = cfg.h_sweeps;
            nmf_solver = std::make_unique<HalsNMF>(hals_cfg);

        } else if (cfg.nmf_type == "mu") {
            MuNMF::Config mu_cfg;
            mu_cfg.n_components  = cfg.n_components;
            mu_cfg.max_iter      = cfg.max_iter;
            mu_cfg.tol           = cfg.tol;
            mu_cfg.verbose       = cfg.debug;
            mu_cfg.random_state  = seed;
            mu_cfg.init_method   = init_enum;
            mu_cfg.acol_p        = cfg.acol_p;
            mu_cfg.compute_error = cfg.debug;
            nmf_solver = std::make_unique<MuNMF>(mu_cfg);

        } else {
            throw std::invalid_argument("Unknown NMF solver: " + cfg.nmf_type);
        }

        // 4. Configure Backend
        std::unique_ptr<IVFBackend> backend;
        std::unique_ptr<IVFBackend::SearchParams> search_params;
        if (cfg.backend_type == "naive") {
            NaiveIVFBackend::Config naive_cfg(cfg.m, true);
            search_params = std::make_unique<NaiveIVFBackend::SearchParams>(cfg.nprobe, 1.0);
            backend = std::make_unique<NaiveIVFBackend>(naive_cfg);
        } else {
            throw std::invalid_argument("Unknown backend type: " + cfg.backend_type);
        }

        // 5. Build Index
        NMFIndex::Config idx_cfg(cfg.sample_size, true);
        NMFIndex index(std::move(backend), idx_cfg);

        index.build(train, std::move(nmf_solver));

        // 6. Save Index (Optional)
        if (!cfg.save_index_path.empty()) {
            index.save_index(cfg.save_index_path);
        }

        // 7. Search & Save Results (Optional)
        if (!cfg.save_results_path.empty()) {

            index.search_and_save(cfg.save_results_path, query, train, cfg.eval_k, search_params.get());
        }

        if (cfg.evaluate_recall) {
            evaluate_nmf_index(index, query, train, gt, {10, 30, 100}, {1,2,3,4,5,10,20,50}, cfg.eval_k, cfg.nprobe);
        }

    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
