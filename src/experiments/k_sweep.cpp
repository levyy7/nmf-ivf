#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <omp.h>
#include <cblas.h>

#include "utils/hdf5_loader.h"
#include "utils/index_eval.h" // For recall_at_k
#include "index/nmf_index.h"
#include "index/nmf/mu_nmf.h"
#include "index/backend/adaptive.h"

// Struct to track the absolute best configuration per K
struct OptimalConfig {
  int k = 0;
  int m = 0;
  float dr = 0.0f;
  double recall30 = 0.0;
  double latency_ms = 999999.0;
  bool found = false;
};

int main() {
  int threads = 8;
  openblas_set_num_threads(threads);
  omp_set_num_threads(threads);

  std::string data_path = "data/nq.h5";

  // The K values you want to test
  std::vector<int> k_values = {1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000,
                               7500, 10000};

  // The Adaptive grid to sweep for each K
  // Pushing the boundaries from Deep Dives to Broad Skims
  // The 20-Point Pareto Frontier Sweep
  std::vector<int> m_values = {50, 75, 100, 125, 150, 175, 200, 250, 300, 350,
                               400, 500};
  std::vector<float> dr_values = {0.5f, 0.40f, 0.35f, 0.30f, 0.25f, 0.20f,
                                  0.15f, 0.10f, 0.05f};

  std::vector<std::pair<int, float>> adapt_grid;
  for (int m : m_values) {
    for (float dr : dr_values) {
      adapt_grid.push_back({m, dr});
    }
  }

  std::cout << "[Setup] Loading dataset from " << data_path << "...\n";
  HDF5Loader loader(data_path);
  auto queries = loader.loadSparse<float>("otest/queries");
  auto gt = loader.loadGroundTruth("otest/knns");
  auto X_train = loader.loadSparse<float>("train");

  std::vector<OptimalConfig> summary;

  for (int k : k_values) {
    std::cout << "\n======================================================\n";
    std::cout << "                TESTING K = " << k << "\n";
    std::cout << "======================================================\n";

    // Step A: Train NMF
    MuNMF::Config mu_cfg;
    mu_cfg.n_components = k;
    mu_cfg.max_iter = 20; // FIXED as requested for fast sweeping
    mu_cfg.verbose = false;
    auto nmf_solver = std::make_unique<MuNMF>(mu_cfg);

    // Step B: Configure Dynamic Backend
    AdaptiveIVFBackend::Config back_cfg;
    auto backend = std::make_unique<AdaptiveIVFBackend>(back_cfg);

    // Step C: Build Index
    NMFIndex::Config idx_cfg(100000, false);
    NMFIndex index(std::move(backend), idx_cfg);

    std::cout << "[Experiment] Fitting NMF and building index...\n";
    index.build(X_train, std::move(nmf_solver));
    index.save_index("output/k_sweep/k" + std::to_string(k) + ".h5");

    // Step D: Sweep Adaptive Parameters
    OptimalConfig optimal;
    optimal.k = k;

    std::cout << "\n[Experiment] Sweeping adaptive parameters...\n";
    std::cout << std::left << std::setw(15) << "Config"
        << std::right << std::setw(12) << "Recall@30"
        << std::setw(15) << "Latency (ms)" << "   Status\n";
    for (int i = 0; i < 55; ++i) std::cout << "-";
    std::cout << "\n";

    for (auto [m, dr] : adapt_grid) {
      AdaptiveIVFBackend::SearchParams params(m, dr);

      auto t0 = std::chrono::steady_clock::now();
      auto results = index.search(queries, X_train, 30, &params);
      auto t1 = std::chrono::steady_clock::now();

      double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).
          count();
      double ms_per_query = total_ms / queries.rows();

      // Calculate Recall@30
      std::vector<double> r30s;
      r30s.reserve(queries.rows());

      for (int qi = 0; qi < queries.rows(); ++qi) {
        std::vector<int> ranked;
        ranked.reserve(results[qi].size());
        for (const auto& r : results[qi]) ranked.push_back(r.id);

        r30s.push_back(recall_at_k(ranked, gt.row(qi), 30));
      }
      double r30 = mean(r30s);

      // Print the run
      char buffer[32];
      snprintf(buffer, sizeof(buffer), "m=%d, dr=%.2f", m, dr);
      std::cout << std::left << std::setw(15) << buffer
          << std::right << std::fixed << std::setprecision(4)
          << std::setw(12) << r30
          << std::setw(15) << ms_per_query;

      // Check if it's the new optimal
      if (r30 >= 0.90) {
        if (ms_per_query < optimal.latency_ms) {
          index.search_and_save("output/k_sweep/k" + std::to_string(k) + ".h5",
                                queries, X_train, 30, &params);
          optimal.m = m;
          optimal.dr = dr;
          optimal.recall30 = r30;
          optimal.latency_ms = ms_per_query;
          optimal.found = true;
          std::cout << "   [NEW OPTIMAL]";
        } else {
          std::cout << "   [Passed 90%]";
        }
      } else {
        std::cout << "   [Failed 90%]";
      }
      std::cout << "\n";
    }

    summary.push_back(optimal);
  }

  // ── 3. Final Pareto Frontier Summary ─────────────────────────────────
  std::cout << "\n\n======================================================\n";
  std::cout << "               FINAL K-SWEEP SUMMARY\n";
  std::cout << "     (Goal: Min Latency where Recall@30 >= 90%)\n";
  std::cout << "======================================================\n";
  std::cout << std::left << std::setw(10) << "K"
      << std::setw(20) << "Optimal Config"
      << std::setw(15) << "Recall@30"
      << std::setw(15) << "Latency (ms)" << "\n";
  for (int i = 0; i < 60; ++i) std::cout << "-";
  std::cout << "\n";

  for (const auto& opt : summary) {
    std::cout << std::left << std::setw(10) << opt.k;

    if (opt.found) {
      char buffer[32];
      snprintf(buffer, sizeof(buffer), "m=%d, dr=%.2f", opt.m, opt.dr);
      std::cout << std::setw(20) << buffer
          << std::fixed << std::setprecision(4) << std::setw(15) << opt.recall30
          << std::setw(15) << opt.latency_ms << "\n";
    } else {
      std::cout << std::setw(20) << "NONE FOUND"
          << std::setw(15) << "---"
          << std::setw(15) << "---" << "\n";
    }
  }
  std::cout << "======================================================\n\n";

  return 0;
}
