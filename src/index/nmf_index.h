#pragma once

#include <vector>

#include "nmf/base.h"

// =====================================================================
// NMFIndex — Soft Inverted Index over MiniBatchNMF Components
// =====================================================================
//
// Builds an approximate nearest-neighbour index using MiniBatchNMF:
//
//   Build:
//     1. Fit MiniBatchNMF to training data X  (n × vocab)
//     2. Project train: W_train = model.transform(X)  shape (n, k)
//     3. For each component r = 0..k-1, keep the top-m documents
//        ranked by W_train[:,r] — the inverted list for component r.
//        "Soft": one document can appear in several lists.
//
//   Search:
//     1. Project query: w_q = model.transform(q)  shape (1, k)
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
//   MiniBatchNMF::Config nmf_cfg;
//   nmf_cfg.n_components = 512;
//   nmf_cfg.batch_size   = 1024;
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
    // ── Configuration ────────────────────────────────────────────────
    struct Config {
        int  m       = 100;   // top-m docs stored per inverted list
        int  nprobe  = 8;     // default lists to probe at query time
        int sample_size = 100000;
        bool verbose = true;
    };

    // ── Result ───────────────────────────────────────────────────────
    struct Result {
        int   doc_id;
        float score;  // w_q · W_train[doc_id,:] in MiniBatchNMF projected space
    };

    // ── Construction ─────────────────────────────────────────────────
    // nmf_cfg controls the inner MiniBatchNMF configuration.
    // idx_cfg controls index parameters (m, nprobe).
    NMFIndex(std::unique_ptr<NMFBase> nmf, const Config& idx_cfg);

    // ── build() ──────────────────────────────────────────────────────
    // Fits the MiniBatchNMF model on X, projects all training vectors, and
    // constructs the k inverted lists.  Safe to call multiple times
    // (model and lists are fully reset each time).
    void build(const SpMat& X);

    // ── search() ─────────────────────────────────────────────────────
    // Projects all queries, probes the top-nprobe lists per query,
    // and returns up to top_k results sorted descending by score.
    //
    // nprobe: if < 0, uses Config::nprobe.
    // Returns: results[qi] for qi = 0 .. queries.rows()-1.
    std::vector<std::vector<Result>> search(
        const SpMat&     queries,
        int              top_k,
        int              nprobe = -1) const;

    // ── Accessors ────────────────────────────────────────────────────
    int                 n_lists()       const { return H_.rows(); }
    int                 n_docs()        const { return n_docs_; }
    bool                is_built()      const { return built_; }
    const Config&       index_config()  const { return cfg_; }

    // Number of documents stored in list r (≤ cfg_.m).
    int list_size(int r) const;

private:
    // One entry in an inverted list.
    struct ListEntry {
        int   doc_id;
        float score;   // W_train[doc_id, r] — component projection weight
    };

    std::unique_ptr<NMFBase>            nmf_;
    Config                              cfg_;

    const SpMat*                        X_docs_;
    Eigen::MatrixXf                     H_;
    int                                 n_docs_ = 0;

    std::vector<std::vector<ListEntry>> lists_;    // (k, ≤m) sorted desc
    bool                                built_  = false;

    void compute_H(const SpMat& X_fit);
    // Build all k inverted lists from projected training matrix W (n × k).
    void build_lists(const SpMat& X);

    // Score and rank candidates for one query vector w_q (size k).
    std::vector<Result> search_one(
        const Eigen::SparseVector<float, Eigen::RowMajor>& query,
        const Eigen::RowVectorXf& query_scores,
        int top_k,
        int nprobe) const;
};