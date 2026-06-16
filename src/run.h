#pragma once

#include <iostream>
#include <omp.h>
#include <Eigen/Sparse>
#include <cblas.h>
#include <optional>
#include <random>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "utils/hdf5_loader.h"
#include "index/nmf_index.h"
#include "index/backend/naive.h"
#include "index/backend/adaptive.h"
#include "index/nmf/hals_nmf.h"
#include "index/nmf/mu_nmf.h"

struct RunConfig {
  std::string input_path;
  std::string task_desc_path;
  std::string output_dir;
  std::string output_path;

  std::string task_id = "task";
  std::string dataset_name = "nq";
  std::string h5_train_path = "train";
  std::string h5_queries_path = "otest/queries";

  int threads = 8;

  // ── NMF ──
  std::string nmf_type = "mu";
  int n_components = 3000;
  double tol = 1e-4;
  int max_iter = 30;
  double forget_factor = 0.7;
  std::string init_method = "acol"; // "acol" | "random"
  int acol_p = 5;
  bool debug = false;
  std::optional<int> random_state = std::nullopt;
  int sample_size = 150000;

  // ── Index & Backend ──
  std::string backend_type = "adaptive";
  int m = 5000;

  // Naive Params
  int nprobe = 20;

  // Adaptive Params
  std::vector<int> max_misses = {80};
  std::vector<float> drop_ratio = {0.20f};

  // ── File I/O ──
  bool skip_save_index = false;
  bool skip_save_results = false;

  // ── Search Config ──
  int eval_k = 30;
};

inline NMFBase::Init parseInitMethod(const std::string& s) {
  if (s == "acol") return NMFBase::Init::Acol;
  if (s == "random") return NMFBase::Init::Random;
  throw std::invalid_argument(
      "Unknown init method: '" + s + "'. Valid: acol, random");
}

inline RunConfig preset(const std::string& name) {
  RunConfig cfg;
  cfg.dataset_name = name;

  if (name == "nq") {
    cfg.n_components = 3000;
    cfg.max_iter = 25;
    cfg.m = 5000;
    cfg.nprobe = 20;
    cfg.max_misses = {60, 80, 100};
    cfg.drop_ratio = {0.10f, 0.20f, 0.30f};
  } else if (name == "fiqa-dev" || name == "fiqa-small") {
    cfg.n_components = 512;
    cfg.max_iter = 50;
    cfg.m = 500;
    cfg.nprobe = 16;
    cfg.max_misses = {20, 30, 40};
    cfg.drop_ratio = {0.10f, 0.15f, 0.20f};
  } else {
    std::cout << "[Preset] No preset for dataset '" << name <<
        "'. Using defaults.\n";
  }
  return cfg;
}

// ── Execution Logic ───────────────────────────────────────────────────────────

inline int run(const RunConfig& cfg) {
  try {
    // 1. Environment Setup
    openblas_set_num_threads(cfg.threads);
    omp_set_num_threads(cfg.threads);
    Eigen::setNbThreads(8);

    const unsigned int seed = cfg.random_state.has_value()
                                ? static_cast<unsigned int>(*cfg.random_state)
                                : std::random_device{}();

    std::cout << "========================================================\n";
    std::cout << " SYSTEM ENVIRONMENT\n";
    std::cout << "========================================================\n";
    std::cout << " Threads          : " << cfg.threads << " (OMP Max: " <<
        omp_get_max_threads() << ")\n";
    std::cout << " Eigen Threads    : " << Eigen::nbThreads() << "\n";
    std::cout << " Seed             : " << seed << "\n\n";

    std::cout << "========================================================\n";
    std::cout << " DATASET & I/O\n";
    std::cout << "========================================================\n";
    std::cout << " Task ID          : " << cfg.task_id << "\n";
    std::cout << " Dataset Name     : " << cfg.dataset_name << "\n";
    std::cout << " Input File       : " << cfg.input_path << "\n";
    std::cout << " Output Base      : " << cfg.output_path << "\n";
    std::cout << " Train Path (H5)  : " << cfg.h5_train_path << "\n";
    std::cout << " Query Path (H5)  : " << cfg.h5_queries_path << "\n\n";

    std::cout << "========================================================\n";
    std::cout << " NMF CONFIGURATION\n";
    std::cout << "========================================================\n";
    std::cout << " Solver           : " << cfg.nmf_type << "\n";
    std::cout << " Components (k)   : " << cfg.n_components << "\n";
    std::cout << " Max Iterations   : " << cfg.max_iter << "\n";
    std::cout << " Tolerance        : " << cfg.tol << "\n";
    std::cout << " Initialization   : " << cfg.init_method << " (p=" << cfg.
        acol_p << ")\n";
    std::cout << " Sample Size      : " << cfg.sample_size << "\n\n";

    std::cout << "========================================================\n";
    std::cout << " INDEX & SEARCH CONFIGURATION\n";
    std::cout << "========================================================\n";
    std::cout << " Backend Type     : " << cfg.backend_type << "\n";
    std::cout << " List Capacity(m) : " << cfg.m << "\n";
    std::cout << " Top-K Saved      : " << cfg.eval_k << "\n\n";

    if (cfg.backend_type == "adaptive") {
      std::cout << " Sweep Misses (m) : ";
      for (int m : cfg.max_misses) std::cout << m << " ";
      std::cout << "\n Sweep Drop (dr)  : ";
      for (float dr : cfg.drop_ratio) std::cout << dr << " ";
      std::cout << "\n";
    } else if (cfg.backend_type == "naive") {
      std::cout << " NProbe           : " << cfg.nprobe << "\n";
    }
    std::cout << "========================================================\n\n";

    // 2. Load Data
    std::cout << "[IO] Loading dataset from " << cfg.input_path << "...\n";
    HDF5Loader loader(cfg.input_path);
    auto query = loader.loadSparse<float>(cfg.h5_queries_path);
    auto train = loader.loadSparse<float>(cfg.h5_train_path);

    std::cout << "[IO] Train matrix: " << train.rows() << " x " << train.cols()
        << " (" << train.nonZeros() << " non-zeros)\n";

    // 3. Configure NMF Solver
    const NMFBase::Init init_enum = parseInitMethod(cfg.init_method);
    std::unique_ptr<NMFBase> nmf_solver;

    if (cfg.nmf_type == "hals") {
      HalsNMF::Config hals_cfg;
      hals_cfg.n_components = cfg.n_components;
      hals_cfg.max_iter = cfg.max_iter;
      hals_cfg.tol = cfg.tol;
      hals_cfg.verbose = cfg.debug;
      hals_cfg.random_state = seed;
      hals_cfg.init_method = init_enum;
      hals_cfg.acol_p = cfg.acol_p;
      hals_cfg.compute_error = cfg.debug;
      nmf_solver = std::make_unique<HalsNMF>(hals_cfg);
    } else if (cfg.nmf_type == "mu") {
      MuNMF::Config mu_cfg;
      mu_cfg.n_components = cfg.n_components;
      mu_cfg.max_iter = cfg.max_iter;
      mu_cfg.tol = cfg.tol;
      mu_cfg.verbose = cfg.debug;
      mu_cfg.random_state = seed;
      mu_cfg.init_method = init_enum;
      mu_cfg.acol_p = cfg.acol_p;
      mu_cfg.compute_error = cfg.debug;
      nmf_solver = std::make_unique<MuNMF>(mu_cfg);
    } else {
      throw std::invalid_argument("Unknown NMF solver: " + cfg.nmf_type);
    }

    // 4. Configure Backend
    std::unique_ptr<IVFBackend> backend;
    if (cfg.backend_type == "naive") {
      NaiveIVFBackend::Config naive_cfg(true);
      backend = std::make_unique<NaiveIVFBackend>(naive_cfg);
    } else if (cfg.backend_type == "adaptive") {
      AdaptiveIVFBackend::Config adapt_cfg(true);
      backend = std::make_unique<AdaptiveIVFBackend>(adapt_cfg);
    } else {
      throw std::invalid_argument("Unknown backend type: " + cfg.backend_type);
    }

    // 5. Build Index
    NMFIndex::Config idx_cfg(cfg.sample_size, true, seed);
    NMFIndex index(std::move(backend), idx_cfg);
    index.build(train, std::move(nmf_solver));

    // 6. Save Index (Optional)
    if (!cfg.skip_save_index) {
      std::cout << "[IO] Saving index to " << cfg.output_path << "\n";
      index.save_index(cfg.output_path);
    }

    // 7. Search & Save Results for Each Parameter Combination
    if (!cfg.skip_save_results) {
      if (cfg.backend_type == "adaptive") {
        // Strip the ".h5" extension to append parameter flags cleanly
        std::string base_out_path = cfg.output_path;
        size_t dot_idx = base_out_path.find_last_of(".");
        if (dot_idx != std::string::npos) {
          base_out_path = base_out_path.substr(0, dot_idx);
        }

        std::cout << "\n[Search] Running sweep over adaptive parameters...\n";
        for (int m : cfg.max_misses) {
          for (float dr : cfg.drop_ratio) {
            char file_suffix[64];
            snprintf(file_suffix, sizeof(file_suffix), "_m%d_dr%.2f.h5", m, dr);
            std::string run_out_path = base_out_path + file_suffix;

            std::cout << "  -> Saving parameters (m=" << m << ", dr=" << dr
                << ") to: " << run_out_path << "\n";

            AdaptiveIVFBackend::SearchParams test_params(m, dr);
            index.search_and_save(run_out_path, query, train, cfg.eval_k,
                                  &test_params);
          }
        }
      } else {
        // Fallback for Naive backend (no sweep configuration applied here)
        std::cout << "\n[Search] Saving results to " << cfg.output_path << "\n";
        NaiveIVFBackend::SearchParams test_params(cfg.nprobe, 1.0f);
        index.search_and_save(cfg.output_path, query, train, cfg.eval_k,
                              &test_params);
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Fatal Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}