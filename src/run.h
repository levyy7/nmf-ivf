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

#include "utils/hdf5_loader.h"

#include "index/nmf_index.h"
#include "index/backend/naive.h"
#include "index/backend/adaptive.h"
#include "index/nmf/hals_nmf.h"
#include "index/nmf/mu_nmf.h"
#include "utils/index_eval.h"

struct RunConfig {
  // ── New Execution Paths ──
  std::string input_path;
  std::string task_desc_path;
  std::string output_dir;
  std::string output_path;

  // ── JSON Extracted Fields ──
  std::string task_id = "task";
  std::string dataset_name = "nq";
  std::string h5_train_path = "train";
  std::string h5_queries_path = "otest/queries";
  std::string h5_gt_path = "otest/knns";

  // ── Environment / Threads ──
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

  // Only for HALS
  int w_sweeps = 1;
  int h_sweeps = 1;

  // ── Index & Backend ──
  std::string backend_type = "adaptive";
  int m = 5000;

  // Naive Params
  int nprobe = 20;

  // Adaptive Params
  int max_misses = 80;
  float drop_ratio = 0.20f;

  // ── File I/O ──
  bool skip_save_index = false;
  bool skip_save_results = false;

  // ── Eval ──
  bool evaluate_recall = false;
  std::vector<int> recall_at = {10, 30, 100};
  int eval_k = 30;
};

// ── Helpers ──────────────────────────────────────────────────────────────────

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
    cfg.max_misses = 80;
    cfg.drop_ratio = 0.20f;
  } else if (name == "fiqa-dev" || name == "fiqa-small") {
    cfg.n_components = 512;
    cfg.max_iter = 50;
    cfg.m = 500;
    cfg.nprobe = 16;
    cfg.max_misses = 30;
    cfg.drop_ratio = 0.15f;
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

    const unsigned int seed = cfg.random_state.has_value()
                                ? static_cast<unsigned int>(*cfg.random_state)
                                : std::random_device{}();

    std::cout << "========================================\n"
        << "Task ID       : " << cfg.task_id << "\n"
        << "Dataset Name  : " << cfg.dataset_name << "\n"
        << "Input File    : " << cfg.input_path << "\n"
        << "Output File   : " << cfg.output_path << "\n"
        << "----------------------------------------\n"
        << "NMF Solver    : " << cfg.nmf_type << "\n"
        << "Components (k): " << cfg.n_components << "\n"
        << "Initialization: " << cfg.init_method << " (p=" << cfg.acol_p <<
        ")\n"
        << "Backend Type  : " << cfg.backend_type << "\n"
        << "List Cap (m)  : " << cfg.m << "\n"
        << "Threads       : " << cfg.threads << " (OMP Max: " <<
        omp_get_max_threads() << ")\n"
        << "Seed          : " << seed << "\n"
        << "========================================\n";

    // 2. Load Data dynamically based on JSON
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
    std::unique_ptr<IVFBackend::SearchParams> search_params;

    if (cfg.backend_type == "naive") {
      NaiveIVFBackend::Config naive_cfg(true);
      search_params = std::make_unique<NaiveIVFBackend::SearchParams>(
          cfg.nprobe, 1.0f);
      backend = std::make_unique<NaiveIVFBackend>(naive_cfg);
    } else if (cfg.backend_type == "adaptive") {
      AdaptiveIVFBackend::Config adapt_cfg(true);
      search_params = std::make_unique<AdaptiveIVFBackend::SearchParams>(
          cfg.max_misses, cfg.drop_ratio);
      backend = std::make_unique<AdaptiveIVFBackend>(adapt_cfg);
    } else {
      throw std::invalid_argument("Unknown backend type: " + cfg.backend_type);
    }

    // 5. Build Index
    NMFIndex::Config idx_cfg(cfg.sample_size, true);
    NMFIndex index(std::move(backend), idx_cfg);

    index.build(train, std::move(nmf_solver));

    // 6. Save Index (Optional)
    if (!cfg.skip_save_index) {
      std::cout << "[IO] Saving index to " << cfg.output_path << "\n";
      index.save_index(cfg.output_path);
    }

    // 7. Search & Save Results (Optional)
    if (!cfg.skip_save_results) {
      std::cout << "[IO] Saving results to " << cfg.output_path << "\n";
      index.search_and_save(cfg.output_path, query, train, cfg.eval_k,
                            search_params.get());
    }

    // 8. Evaluation & Parameter Sweep
    if (cfg.evaluate_recall) {
      Eigen::MatrixXi gt;
      bool has_gt = true;
      try {
        gt = loader.loadGroundTruth(cfg.h5_gt_path);
      } catch (...) {
        std::cout << "[Eval] Warning: Cannot load GT from '" << cfg.h5_gt_path
            << "'. Skipping evaluation step.\n";
        has_gt = false;
      }

      if (has_gt) {
        std::vector<std::pair<std::string, std::unique_ptr<
                                IVFBackend::SearchParams>>> sweep_configs;
        std::string default_config_name;

        if (cfg.backend_type == "naive") {
          default_config_name = "nprobe=" + std::to_string(cfg.nprobe);
          for (int np : {1, 2, 5, 10, 20, 50}) {
            std::string name = "np=" + std::to_string(np);
            sweep_configs.push_back(
            {name,
             std::make_unique<NaiveIVFBackend::SearchParams>(np, 1.0f)});
          }
        } else if (cfg.backend_type == "adaptive") {
          default_config_name = "m=" + std::to_string(cfg.max_misses) + ", dr="
                                +
                                std::to_string(cfg.drop_ratio);

          std::vector<std::pair<int, float>> adapt_sweep = {
              {60, 0.30f}, {80, 0.30f}, {80, 0.20f}, {100, 0.20f}, {120, 0.15f},
              {100, 0.10f}
          };
          for (auto [m, dr] : adapt_sweep) {
            char buffer[32];
            snprintf(buffer, sizeof(buffer), "m=%d, dr=%.2f", m, dr);
            sweep_configs.push_back(
            {buffer,
             std::make_unique<AdaptiveIVFBackend::SearchParams>(m, dr)});
          }
        }

        std::vector<std::pair<std::string, const IVFBackend::SearchParams*>>
            sweep_ptrs;
        for (const auto& conf : sweep_configs) {
          sweep_ptrs.push_back({conf.first, conf.second.get()});
        }

        evaluate_nmf_index(index, query, train, gt, cfg.recall_at, cfg.eval_k,
                           default_config_name, search_params.get(),
                           sweep_ptrs);
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Fatal Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}