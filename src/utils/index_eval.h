// experiments/nmf_evaluation.h
#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <unordered_set>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <string>
#include <limits>

#include <Eigen/Dense>

#include "index/nmf_index.h"

// ============================================================
// Helpers
// ============================================================

inline void hr(char c = '=') {
  for (int i = 0; i < 74; ++i) std::cout << c;
  std::cout << '\n';
}

inline void section(const std::string& s) {
    std::cout << '\n'; hr('='); std::cout << s << '\n'; hr('=');
}

inline void subsection(const std::string& s) {
    std::cout << '\n'; std::cout << s << '\n'; hr('-');
}

template<typename T>
inline void kv(const std::string& k, const T& v) {
    std::cout << std::left << std::setw(24) << k << v << '\n';
}

inline double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

inline double stddev(const std::vector<double>& v) {
    if (v.size() <= 1) return 0.0;
    const double m = mean(v);
    double acc = 0.0;
    for (double x : v) {
        const double d = x - m;
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(v.size()));
}

// ============================================================
// Recall@k (Normalized 0 to 1)
// ============================================================

inline double recall_at_k(const std::vector<int>& ranked,
                          const Eigen::VectorXi& gt_row,
                          int k)
{
    std::unordered_set<int> gt;
    for (int i = 0; i < gt_row.size(); ++i) {
        if (gt_row[i] >= 0) gt.insert(gt_row[i]);
    }

    if (gt.empty()) return 0.0;

    int hits = 0;
    const int lim = std::min<int>(k, static_cast<int>(ranked.size()));
    for (int i = 0; i < lim; ++i) {
        if (gt.contains(ranked[i])) ++hits;
    }

    const int max_possible_hits = std::min<int>(k, static_cast<int>(gt.size()));
    return static_cast<double>(hits) / static_cast<double>(max_possible_hits);
}

// ============================================================
// Evaluation
// ============================================================

template<typename SparseMat>
void evaluate_nmf_index(
    const NMFIndex& index,
    const SparseMat& queries,
    const SparseMat& X_docs,
    const Eigen::MatrixXi& gt,
    const std::vector<int>& recall_ats,
    int top_k,
    const std::string& default_param_name,
    const IVFBackend::SearchParams* default_params,
    const std::vector<std::pair<std::string, const IVFBackend::SearchParams*>>& param_sweep
) {
  using Clock = std::chrono::steady_clock;

  section("RESULTS");

    subsection("Configuration");
    kv("Queries", queries.rows());
    kv("Vocabulary", queries.cols());
    kv("Latent dimensions (k)", index.backend().n_lists());
    kv("Docs/list (m)", index.lists().cols());
    kv("Indexed docs", X_docs.rows());
    kv("Default config", default_param_name);

    subsection("Search Evaluation");
    std::vector<std::vector<double>> recalls(recall_ats.size());

    auto global_t0 = Clock::now();
    auto batch_results = index.search(queries, X_docs, top_k, default_params);
    auto global_t1 = Clock::now();

    double total_sec = std::chrono::duration<double>(global_t1 - global_t0).count();
    double avg_query_ms = (total_sec * 1000.0) / static_cast<double>(queries.rows());
    double default_qps = queries.rows() / total_sec;

  for (int qi = 0; qi < queries.rows(); ++qi) {
        const auto& results = batch_results[qi];
        std::vector<int> ranked;
        ranked.reserve(results.size());
        for (const auto& r : results) ranked.push_back(r.id);

        for (size_t r = 0; r < recall_ats.size(); ++r) {
            recalls[r].push_back(recall_at_k(ranked, gt.row(qi), recall_ats[r]));
        }
    }

    kv("Total query time", total_sec);
    kv("Avg/query (ms)", avg_query_ms);
    kv("QPS", default_qps);

  subsection("Recall");
    for (size_t r = 0; r < recall_ats.size(); ++r) {
        const double m = mean(recalls[r]);
        const double s = stddev(recalls[r]);
        std::cout << "Recall@" << std::setw(4) << recall_ats[r] << "  "
                  << std::fixed << std::setprecision(4) << m << " ± " << s << "  ";

        int bars = static_cast<int>(m * 40.0);
        for (int i = 0; i < bars; ++i) std::cout << "█";
        std::cout << '\n';
    }

    // --- Dynamic Table Header ---
  section("PARAMETER SWEEP");
    std::cout << std::left << std::setw(18) << "Config";
  for (int k : recall_ats) {
    std::cout << std::right << std::setw(12) << ("R@" + std::to_string(k));
  }
  std::cout << std::setw(14) << "ms/query" << '\n';
  hr('-');

  // --- Tracking Structures for the Pareto Frontier ---
  const std::vector<double> targets = {0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96,
                                       0.97, 0.98, 0.99};

  struct OptRun {
    std::string name = "None";
    double actual_recall = 0.0;
    double ms = std::numeric_limits<double>::max();
    double qps = 0.0;
  };

  // Matrix of OptRun: rows = recall_ats index, cols = targets index
  std::vector<std::vector<OptRun>> best_runs(recall_ats.size(),
                                             std::vector<
                                               OptRun>(targets.size()));

  // --- Run the Sweep ---
  for (const auto& [name, params] : param_sweep) {
        auto t0 = Clock::now();
        auto sweep_results = index.search(queries, X_docs, top_k, params);
        auto t1 = Clock::now();

        double sweep_total_sec = std::chrono::duration<double>(t1 - t0).count();
        double ms_per_query = (sweep_total_sec * 1000.0) / static_cast<double>(
                                queries.rows());
        double qps = static_cast<double>(queries.rows()) / sweep_total_sec;

        std::vector<std::vector<double>> sweep_recalls(recall_ats.size());

        for (int qi = 0; qi < queries.rows(); ++qi) {
            const auto& results = sweep_results[qi];
            std::vector<int> ranked;
            ranked.reserve(results.size());
            for (const auto& r : results) ranked.push_back(r.id);

            for (size_t r = 0; r < recall_ats.size(); ++r) {
              sweep_recalls[r].push_back(
                  recall_at_k(ranked, gt.row(qi), recall_ats[r]));
            }
        }

        std::cout << std::left << std::setw(18) << name;

        // Print and Track
        for (size_t r = 0; r < recall_ats.size(); ++r) {
          double mean_r = mean(sweep_recalls[r]);
          std::cout << std::right << std::fixed << std::setprecision(4) <<
              std::setw(12) << mean_r;

          // Update optimal threshold trackers
          for (size_t t = 0; t < targets.size(); ++t) {
            if (mean_r >= targets[t] && ms_per_query < best_runs[r][t].ms) {
              best_runs[r][t] = {name, mean_r, ms_per_query, qps};
            }
          }
        }
        std::cout << std::setw(14) << ms_per_query << '\n';
  }

  // --- Print Optimal Leaderboards ---
  section("OPTIMAL CONFIGURATIONS PER RECALL THRESHOLD");

  for (size_t r = 0; r < recall_ats.size(); ++r) {
    subsection("Thresholds for Recall@" + std::to_string(recall_ats[r]));
    std::cout << std::left << std::setw(10) << "Target"
        << std::setw(20) << "Best Config"
        << std::right << std::setw(12) << "Actual"
        << std::setw(15) << "Latency (ms)"
        << std::setw(15) << "Throughput" << '\n';
    hr('-');

    for (size_t t = 0; t < targets.size(); ++t) {
      std::cout << std::left << std::setw(10) << (std::to_string(
                                                      static_cast<int>(
                                                        targets[t] * 100)) +
                                                  "%")
          << std::setw(20) << best_runs[r][t].name;

      if (best_runs[r][t].name != "None") {
        std::cout << std::right << std::fixed << std::setprecision(4) <<
            std::setw(12) << best_runs[r][t].actual_recall
            << std::setw(15) << best_runs[r][t].ms
            << std::setw(11) << std::setprecision(0) << best_runs[r][t].qps <<
            " QPS\n";
      } else {
        std::cout << std::right << std::setw(12) << "---"
            << std::setw(15) << "---"
                          << std::setw(15) << "---" << '\n';
            }
        }
    }
    std::cout << '\n';
}