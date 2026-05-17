#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <unordered_set>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "nmf_index.h"

// ============================================================
// Helpers
// ============================================================

inline void hr(char c = '=') {
    for (int i = 0; i < 60; ++i)
        std::cout << c;
    std::cout << '\n';
}

inline void section(const std::string& s) {
    std::cout << '\n';
    hr('=');
    std::cout << s << '\n';
    hr('=');
}

inline void subsection(const std::string& s) {
    std::cout << '\n';
    std::cout << s << '\n';
    hr('-');
}

template<typename T>
inline void kv(const std::string& k, const T& v) {
    std::cout << std::left
              << std::setw(24)
              << k
              << v
              << '\n';
}

inline double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;

    return std::accumulate(v.begin(), v.end(), 0.0) /
           static_cast<double>(v.size());
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
// Recall@k
// ============================================================

inline double recall_at_k(const std::vector<int>& ranked,
                          const Eigen::VectorXi& gt_row,
                          int k)
{
    std::unordered_set<int> gt;

    for (int i = 0; i < gt_row.size(); ++i) {
        if (gt_row[i] >= 0)
            gt.insert(gt_row[i]);
    }

    if (gt.empty())
        return 0.0;

    int hits = 0;

    const int lim = std::min<int>(k, ranked.size());

    for (int i = 0; i < lim; ++i) {
        if (gt.contains(ranked[i]))
            ++hits;
    }

    return static_cast<double>(hits) /
           static_cast<double>(gt.size());
}

// ============================================================
// Evaluation
// ============================================================

template<typename SparseMat>
void evaluate_nmf_index(
    const NMFIndex& index,
    const SparseMat& queries,
    const Eigen::MatrixXi& gt,
    const std::vector<int>& recall_ats = {10, 30, 100},
    const std::vector<int>& nprobe_sweep = {1,2,5,10,20,50},
    int top_k = 100
) {
    using Clock = std::chrono::high_resolution_clock;

    section("RESULTS");

    // ========================================================
    // Configuration
    // ========================================================

    subsection("Configuration");

    kv("Queries", queries.rows());
    kv("Vocabulary", queries.cols());
    kv("Latent dimensions", index.n_lists());
    kv("Docs/list", index.index_config().m);
    kv("Default nprobe", index.index_config().nprobe);
    kv("Indexed docs", index.n_docs());

    // ========================================================
    // Main search evaluation
    // ========================================================

    subsection("Search Evaluation");

    std::vector<std::vector<double>> recalls(
        recall_ats.size()
    );

    auto global_t0 = Clock::now();

    // --------------------------------------------------------
    // Run ALL queries simultaneously
    // --------------------------------------------------------

    auto batch_results = index.search(
        queries,
        top_k
    );

    auto global_t1 = Clock::now();

    double total_sec =
        std::chrono::duration<double>(
            global_t1 - global_t0
        ).count();

    double avg_query_ms =
        (total_sec * 1000.0) /
        static_cast<double>(queries.rows());

    // --------------------------------------------------------
    // Evaluate recalls
    // --------------------------------------------------------

    for (int qi = 0; qi < queries.rows(); ++qi) {

        const auto& results = batch_results[qi];

        std::vector<int> ranked;
        ranked.reserve(results.size());

        for (const auto& r : results)
            ranked.push_back(r.doc_id);

        for (size_t r = 0; r < recall_ats.size(); ++r) {
            recalls[r].push_back(
                recall_at_k(
                    ranked,
                    gt.row(qi),
                    recall_ats[r]
                )
            );
        }
    }

    kv("Total query time", total_sec);
    kv("Avg/query (ms)", avg_query_ms);
    kv("QPS", queries.rows() / total_sec);

    // ========================================================
    // Recall
    // ========================================================

    subsection("Recall");

    for (size_t r = 0; r < recall_ats.size(); ++r) {

        const double m = mean(recalls[r]);
        const double s = stddev(recalls[r]);

        std::cout
            << "Recall@"
            << std::setw(4)
            << recall_ats[r]
            << "  "
            << std::fixed
            << std::setprecision(4)
            << m
            << " ± "
            << s
            << "  ";

        int bars = static_cast<int>(m * 40.0);

        for (int i = 0; i < bars; ++i)
            std::cout << "█";

        std::cout << '\n';
    }

    // ========================================================
    // NPROBE SWEEP
    // ========================================================

    section("NPROBE SWEEP");

    std::cout
        << std::setw(10) << "nprobe"
        << std::setw(14) << "R@10"
        << std::setw(14) << "R@30"
        << std::setw(14) << "R@100"
        << std::setw(14) << "ms/query"
        << '\n';

    hr('-');

    for (int nprobe : nprobe_sweep) {

        auto t0 = Clock::now();

        auto batch_results = index.search(
            queries,
            top_k,
            nprobe
        );

        auto t1 = Clock::now();

        double total_ms =
            std::chrono::duration<double, std::milli>(
                t1 - t0
            ).count();

        double ms_per_query =
            total_ms / static_cast<double>(queries.rows());

        std::vector<double> r10s;
        std::vector<double> r30s;
        std::vector<double> r100s;

        for (int qi = 0; qi < queries.rows(); ++qi) {

            const auto& results = batch_results[qi];

            std::vector<int> ranked;
            ranked.reserve(results.size());

            for (const auto& r : results)
                ranked.push_back(r.doc_id);

            r10s.push_back(
                recall_at_k(
                    ranked,
                    gt.row(qi),
                    10
                )
            );

            r30s.push_back(
                recall_at_k(
                    ranked,
                    gt.row(qi),
                    30
                )
            );

            r100s.push_back(
                recall_at_k(
                    ranked,
                    gt.row(qi),
                    100
                )
            );
        }

        std::cout
            << std::setw(10) << nprobe
            << std::setw(14) << std::fixed << std::setprecision(4) << mean(r10s)
            << std::setw(14) << mean(r30s)
            << std::setw(14) << mean(r100s)
            << std::setw(14) << ms_per_query;

        if (nprobe == index.n_lists())
            std::cout << "  <- full index";

        std::cout << '\n';
    }

    std::cout << '\n';
}