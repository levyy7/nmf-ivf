#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <ostream>
#include <sstream>
#include <unordered_set>
#include <vector>
#include <memory>
#include <omp.h>

#include "index/nmf_index.h"
#include "index/backend/naive.h"
#include "utils/hdf5_loader.h"

// ─────────────────────────────────────────────────────────────────────────────
// analyse_knn_coverage
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int KNN_ANALYSIS_N_BINS = 10;

struct KNNListStats {
    int n_ranks;
    int n_queries;
    int n_neighbors;   // nn per query

    std::vector<double>                              avg_found;      // [rank]
    std::vector<double>                              cumul_recall;   // [rank]
    std::vector<double>                              mean_rel_pos;   // [rank]
    std::vector<double>                              std_rel_pos;    // [rank]
    std::vector<std::array<int, KNN_ANALYSIS_N_BINS>> pos_hist;      // [rank][bin]

    double overall_recall;
};

inline KNNListStats analyse_knn_coverage(
        const NMFIndex&        index,
        const SpMat&           queries,
        const Eigen::MatrixXi& gt,
        int                    n_ranks = 20,
        int                    nn = 30)
{
    if (!index.is_built())
        throw std::runtime_error("analyse_knn_coverage: index not built");

    const int Q  = static_cast<int>(queries.rows());
    const int k  = index.backend().n_lists();
    n_ranks      = std::min(n_ranks, k);

    // Access the lists directly as a k x m Eigen Matrix of document IDs
    const Eigen::MatrixXi& lists = index.lists();

    // Project all queries at once: (Q × vocab) × (vocab × k) → Q × k
    const Eigen::MatrixXf projected = queries * index.components().transpose();

    std::vector<int>                    found_count(n_ranks, 0);
    std::vector<std::vector<double>>    rel_pos_samples(n_ranks);
    int                                 total_found = 0;

    std::vector<int> ranked(k);   // reused per query

    // Determine the actual maximum ground truth neighbors available
    const int actual_nn = std::min(nn, static_cast<int>(gt.cols()));

    for (int qi = 0; qi < Q; ++qi) {

        const Eigen::RowVectorXf& qscores = projected.row(qi);

        // Rank components by query–component score, descending.
        std::iota(ranked.begin(), ranked.end(), 0);
        std::sort(ranked.begin(), ranked.end(), [&](int a, int b) {
            return qscores[a] > qscores[b];
        });

        // GT set for this query.
        std::unordered_set<int> gt_set;
        gt_set.reserve(actual_nn * 2);
        for (int j = 0; j < actual_nn; ++j) {
            int gt_doc = gt(qi, j);
            if (gt_doc >= 0) gt_set.insert(gt_doc);
        }

        // Exclusive assignment: track which neighbours are already placed.
        std::unordered_set<int> assigned;
        assigned.reserve(actual_nn * 2);

        for (int r = 0; r < n_ranks; ++r) {
            if (static_cast<int>(assigned.size()) == actual_nn) break;

            const int topic = ranked[r];
            const int list_size = index.backend().list_size(topic);

            if (list_size == 0) continue;

            for (int pos = 0; pos < list_size; ++pos) {
                const int doc = lists(topic, pos);

                if (doc < 0) break; // Reached padding
                if (!gt_set.count(doc) || assigned.count(doc)) continue;

                assigned.insert(doc);
                ++found_count[r];

                rel_pos_samples[r].push_back(static_cast<double>(pos) / list_size);
            }
        }

        total_found += static_cast<int>(assigned.size());
    }

    // ── Assemble result struct ────────────────────────────────────────
    KNNListStats s;
    s.n_ranks     = n_ranks;
    s.n_queries   = Q;
    s.n_neighbors = actual_nn;
    s.avg_found   .resize(n_ranks, 0.0);
    s.cumul_recall.resize(n_ranks, 0.0);
    s.mean_rel_pos.resize(n_ranks, 0.0);
    s.std_rel_pos .resize(n_ranks, 0.0);
    s.pos_hist    .resize(n_ranks);
    for (auto& h : s.pos_hist) h.fill(0);

    for (int r = 0; r < n_ranks; ++r) {
        s.avg_found[r] = static_cast<double>(found_count[r]) / Q;

        const auto& samples = rel_pos_samples[r];
        if (samples.empty()) continue;

        double mean = 0.0;
        for (double v : samples) mean += v;
        mean /= static_cast<double>(samples.size());
        s.mean_rel_pos[r] = mean;

        double var = 0.0;
        for (double v : samples) var += (v - mean) * (v - mean);
        s.std_rel_pos[r] = std::sqrt(var / static_cast<double>(samples.size()));

        for (double v : samples) {
            int bin = std::min(
                static_cast<int>(v * KNN_ANALYSIS_N_BINS),
                KNN_ANALYSIS_N_BINS - 1);
            ++s.pos_hist[r][bin];
        }
    }

    double cumul = 0.0;
    for (int r = 0; r < n_ranks; ++r) {
        cumul += s.avg_found[r];
        s.cumul_recall[r] = cumul / actual_nn;
    }

    s.overall_recall =
        static_cast<double>(total_found) / (static_cast<double>(Q) * actual_nn);

    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// print_knn_coverage
// ─────────────────────────────────────────────────────────────────────────────

inline void print_knn_coverage(const KNNListStats& s,
                                std::ostream& out = std::cout)
{
    // UTF-8 block characters, 6 levels: none → full.
    static constexpr const char* kBlocks[] = {"·", "▁", "▃", "▅", "▇", "█"};
    static constexpr int         kLevels   = 6;

    out << "\n════════════════════════════════════════════════════════════════════\n";
    out << "  KNN List Coverage"
        << "  |  queries=" << s.n_queries
        << "  nn=" << s.n_neighbors
        << "  ranks=" << s.n_ranks << "\n";
    out << std::fixed << std::setprecision(1);
    out << "  Overall recall: " << s.overall_recall * 100.0 << "%\n";
    out << "════════════════════════════════════════════════════════════════════\n";
    out << std::setw(5)  << "Rank"
        << std::setw(11) << "Avg Found"
        << std::setw(14) << "Cumul Recall"
        << "   Rel Pos Mean±Std"
        << "   (front→back, each char=10% of list)\n";
    out << std::string(80, '-') << "\n";

    for (int r = 0; r < s.n_ranks; ++r) {
        // Stop once recall has saturated and no more hits are coming.
        if (r > 0 && s.avg_found[r] < 1e-9 && s.cumul_recall[r - 1] > 0.9999)
            break;

        // Build sparkline bar: one character per decile bin.
        int total = 0;
        for (int b = 0; b < KNN_ANALYSIS_N_BINS; ++b) total += s.pos_hist[r][b];

        std::string bar;
        for (int b = 0; b < KNN_ANALYSIS_N_BINS; ++b) {
            double frac = total > 0
                ? static_cast<double>(s.pos_hist[r][b]) / total : 0.0;
            int level = std::min(
                static_cast<int>(frac * kLevels * KNN_ANALYSIS_N_BINS),
                kLevels - 1);
            bar += kBlocks[level];
        }

        out << std::setw(4) << (r + 1) << "  "
            << std::setw(9) << std::setprecision(3) << s.avg_found[r]
            << "  "
            << std::setw(10) << std::setprecision(1) << (s.cumul_recall[r] * 100.0) << "%";

        if (total > 0) {
            out << "   " << std::setw(6) << std::setprecision(3) << s.mean_rel_pos[r]
                << " ± " << std::setw(5) << std::setprecision(3) << s.std_rel_pos[r];
        } else {
            out << "              —     ";
        }

        out << "   [" << bar << "]\n";
    }

    out << std::string(80, '-') << "\n\n";

    // ── Numeric decile table ──────────────────────────────────────────
    out << "  Position breakdown per rank (% of hits in each decile of list depth)\n";
    out << "  " << std::string(76, '-') << "\n";
    out << "  " << std::setw(5) << "Rank";
    for (int b = 0; b < KNN_ANALYSIS_N_BINS; ++b) {
        std::ostringstream lbl;
        lbl << b * 10 << '-' << (b + 1) * 10 << '%';
        out << std::setw(9) << lbl.str();
    }
    out << "\n  " << std::string(76, '-') << "\n";

    for (int r = 0; r < s.n_ranks; ++r) {
        int total = 0;
        for (int b = 0; b < KNN_ANALYSIS_N_BINS; ++b) total += s.pos_hist[r][b];
        if (total == 0) continue;

        out << "  " << std::setw(5) << (r + 1);
        for (int b = 0; b < KNN_ANALYSIS_N_BINS; ++b) {
            double pct = static_cast<double>(s.pos_hist[r][b]) / total * 100.0;
            out << std::setw(8) << std::setprecision(1) << std::fixed << pct << '%';
        }
        out << "\n";
    }
    out << "\n";
}

int main()
{
    omp_set_num_threads(8);

    HDF5Loader loader("data/nq.h5");

    auto query = loader.loadSparse<float>("otest/queries");
    auto gt    = loader.loadGroundTruth("otest/knns");

    auto factory = [](const Eigen::MatrixXf& H, const Eigen::MatrixXi& lists) {
        return std::make_unique<NaiveIVFBackend>(H, lists);
    };

    std::unique_ptr<NMFIndex> index = NMFIndex::load("output/nmf_index.data", factory);

    KNNListStats stats = analyse_knn_coverage(*index, query, gt, 20, 30);
    print_knn_coverage(stats);

    return 0;
}