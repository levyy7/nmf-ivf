#pragma once

#include "nmf.h"
#include <vector>

// =====================================================================
// NMFIndex — Soft Inverted Index over NMF Components
// =====================================================================
//
// Builds an approximate nearest-neighbour index using NMF:
//
//   Build:
//     1. Fit NMFModel to training data X  (n × vocab)
//     2. Project train: W_train = max(X H^T, 0)  shape (n, k)
//     3. For each component r = 0..k-1, keep the top-m documents
//        ranked by W_train[:,r] — the inverted list for component r.
//        "Soft": one document can appear in several lists.
//
//   Search:
//     1. Project query: w_q = max(q H^T, 0)  shape (1, k)
//     2. Rank all k lists by w_q[r]; probe the top-nprobe lists.
//     3. Collect unique candidate doc_ids from those lists.
//     4. Re-score each candidate: score = w_q · W_train[doc,:]
//     5. Return top-k by score.
//
// Re-scoring in the projected W-space is O(candidates × k) where
// candidates ≤ nprobe × m.  At k=512, m=100, nprobe=16: ≤1600 dot
// products of size 512 — a few hundred microseconds per query.
//
// ── Memory ───────────────────────────────────────────────────────────
//   W_train_: n × k × 4 B  (e.g. 57k × 512 → ~118 MB)
//   lists_  : k × m × 8 B  (e.g. 512 × 100 → ~0.4 MB)
//
// ── Parallelism ──────────────────────────────────────────────────────
//   build()  — OMP parallel-for over k components (list construction)
//   search() — OMP parallel-for over queries; each query independent
//
// ── Usage ────────────────────────────────────────────────────────────
//   NMFModel::Config nmf_cfg;
//   nmf_cfg.k      = 512;
//   nmf_cfg.solver = NMFModel::SolverType::SGD;
//
//   NMFIndex::Config idx_cfg;
//   idx_cfg.m      = 200;   // docs per list
//   idx_cfg.nprobe = 16;    // lists to probe
//
//   NMFIndex index(nmf_cfg, idx_cfg);
//   index.build(X_train);                             // fit + index
//   auto hits = index.search(X_queries, /*top_k=*/10);
//   // hits[qi] → vector of up to 10 Result{doc_id, score}, sorted desc

class NMFIndex {
public:
    // ── Types ────────────────────────────────────────────────────────
    using SparseMat = NMFModel::SparseMat;
    using DenseMat  = NMFModel::DenseMat;

    // Row-major matrix for W_train — keeps W_train_.row(i) contiguous.
    using RowMat = Eigen::Matrix<float,
                                  Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>;

    // ── Configuration ────────────────────────────────────────────────
    struct Config {
        int  m       = 100;   // top-m docs stored per inverted list
        int  nprobe  = 8;     // default lists to probe at query time
        bool verbose = true;
    };

    // ── Result ───────────────────────────────────────────────────────
    struct Result {
        int   doc_id;
        float score;  // w_q · W_train[doc_id,:] in NMF projected space
    };

    // ── Construction ─────────────────────────────────────────────────
    // nmf_cfg controls the inner NMFModel (k, solver, lr, …).
    // idx_cfg controls index parameters (m, nprobe).
    NMFIndex(const NMFModel::Config& nmf_cfg, const Config& idx_cfg);

    // ── build() ──────────────────────────────────────────────────────
    // Fits the NMF model on X, projects all training vectors, and
    // constructs the k inverted lists.  Safe to call multiple times
    // (model and lists are fully reset each time).
    void build(const SparseMat& X);

    // ── search() ─────────────────────────────────────────────────────
    // Projects all queries, probes the top-nprobe lists per query,
    // and returns up to top_k results sorted descending by score.
    //
    // nprobe: if < 0, uses Config::nprobe.
    // Returns: results[qi] for qi = 0 .. queries.rows()-1.
    std::vector<std::vector<Result>> search(
        const SparseMat& queries,
        int              top_k,
        int              nprobe = -1) const;

    // ── Accessors ────────────────────────────────────────────────────
    const NMFModel& model()         const { return model_; }
    int             n_lists()       const { return model_.k(); }
    int             n_docs()        const { return n_docs_; }
    bool            is_built()      const { return built_; }
    const Config&   index_config()  const { return cfg_; }

    // Number of documents stored in list r (≤ cfg_.m).
    int list_size(int r) const;

private:
    // One entry in an inverted list.
    struct ListEntry {
        int   doc_id;
        float score;   // W_train[doc_id, r] — component projection weight
    };

    NMFModel::Config                    nmf_cfg_;
    Config                              cfg_;
    NMFModel                            model_;
    const SparseMat*                    X_docs_;

    std::vector<std::vector<ListEntry>> lists_;    // (k, ≤m) sorted desc
    RowMat                              W_train_;  // (n_docs, k) row-major
    int                                 n_docs_ = 0;
    bool                                built_  = false;

    // Build all k inverted lists from projected training matrix W (n × k).
    // W is passed column-major so W.col(r) reads are contiguous.
    void build_lists(const DenseMat& W);

    // Score and rank candidates for one query vector w_q (size k).
    std::vector<Result> search_one(
        Eigen::Ref<const Eigen::RowVectorXf> w_q,
        int top_k,
        int nprobe) const;
};