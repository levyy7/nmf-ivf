#include "nmf_index.h"

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <omp.h>

// ============================================================
// Constructor
// ============================================================

NMFIndex::NMFIndex(const NMFModel::Config& nmf_cfg,
                   const Config& idx_cfg)
    : nmf_cfg_(nmf_cfg),
      cfg_(idx_cfg),
      model_(nmf_cfg),
      X_docs_(nullptr) {}


// ============================================================
// build()
// ============================================================

void NMFIndex::build(const SparseMat& X) {
    if (cfg_.verbose) {
        std::cout << "NMFIndex::build()\n";
    }
    X_docs_ = &X;

    // 1. Fit NMF
    model_.fit(X);

    const auto k = model_.k();
    n_docs_ = static_cast<int>(X.rows());

    // 2. Project training data ONLY via XH^T
    W_train_ = model_.project(X);   // (n × k), soft assignments

    // 3. Build inverted lists
    lists_.assign(k, {});
    build_lists(W_train_);

    built_ = true;

    if (cfg_.verbose) {
        std::cout << "Index built: n_docs=" << n_docs_
                  << " k=" << k << "\n";
    }
}


// ============================================================
// build_lists()
// ============================================================

void NMFIndex::build_lists(const DenseMat& W) {
    const int k = static_cast<int>(W.cols());
    const int n = static_cast<int>(W.rows());

    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < k; ++r) {

        std::vector<ListEntry> tmp;
        tmp.reserve(n);

        for (int i = 0; i < n; ++i) {
            float score = W(i, r);
            if (score > 0.f) {
                tmp.push_back({i, score});
            }
        }

        // keep top-m
        int m = cfg_.m;
        if (tmp.size() > static_cast<size_t>(m)) {
            std::nth_element(
                tmp.begin(),
                tmp.begin() + m,
                tmp.end(),
                [](const ListEntry& a, const ListEntry& b) {
                    return a.score > b.score;
                }
            );
            tmp.resize(m);
        }

        std::sort(tmp.begin(), tmp.end(),
                  [](const ListEntry& a, const ListEntry& b) {
                      return a.score > b.score;
                  });

        lists_[r] = std::move(tmp);
    }
}


// ============================================================
// list_size()
// ============================================================

int NMFIndex::list_size(int r) const {
    if (r < 0 || r >= static_cast<int>(lists_.size()))
        return 0;
    return static_cast<int>(lists_[r].size());
}


// ============================================================
// search()
// ============================================================

std::vector<std::vector<NMFIndex::Result>>
NMFIndex::search(const SparseMat& queries,
                 int top_k,
                 int nprobe) const
{
    if (!built_)
        throw std::runtime_error("Index not built");

    if (nprobe < 0) nprobe = cfg_.nprobe;

    const int Q = static_cast<int>(queries.rows());

    std::vector<std::vector<Result>> results(Q);

    #pragma omp parallel for schedule(dynamic)
    for (int qi = 0; qi < Q; ++qi) {
        Eigen::RowVectorXf x = Eigen::RowVectorXf::Zero(queries.cols());

        for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(queries, qi); it; ++it) {
            x[it.col()] = it.value();
        }

        results[qi] = search_one(x, top_k, nprobe);
    }

    return results;
}


// ============================================================
// search_one()
// ============================================================

std::vector<NMFIndex::Result>
NMFIndex::search_one(Eigen::Ref<const Eigen::RowVectorXf> x,
                     int top_k,
                     int nprobe) const
{
    // ============================================================
    // 1. Project query ONLY via xH^T
    // ============================================================

    NMFModel::DenseMat X(1, x.size());
    X.row(0) = x;

    Eigen::RowVectorXf w_q = model_.project(X).row(0);

    const int k = static_cast<int>(w_q.size());

    // ============================================================
    // 2. Pick top-nprobe latent components
    // ============================================================

    std::vector<int> comps(k);
    std::iota(comps.begin(), comps.end(), 0);

    const int probes = std::min(nprobe, k);

    std::partial_sort(
        comps.begin(),
        comps.begin() + probes,
        comps.end(),
        [&](int a, int b) {
            return w_q[a] > w_q[b];
        }
    );

    // ============================================================
    // 3. Gather candidates from inverted lists
    // ============================================================

    std::vector<int> candidates;
    candidates.reserve(cfg_.m * probes);

    for (int i = 0; i < probes; ++i) {
        const int r = comps[i];

        for (const auto& e : lists_[r]) {
            candidates.push_back(e.doc_id);
        }
    }

    // ============================================================
    // 4. Deduplicate candidates
    // ============================================================

    std::sort(candidates.begin(), candidates.end());

    candidates.erase(
        std::unique(candidates.begin(), candidates.end()),
        candidates.end()
    );

    // ============================================================
    // 5. FINAL RERANK IN ORIGINAL SPACE
    //
    // score(q, d) = q · x_d
    //
    // q  : dense query vector
    // x_d: sparse document row
    // ============================================================

    std::vector<Result> scored;
    scored.reserve(candidates.size());

    for (int doc : candidates) {

        float score = 0.f;

        // Sparse dot product:
        // iterate only over non-zeros of document row
        for (SparseMat::InnerIterator it(*X_docs_, doc); it; ++it) {
            score += x[it.col()] * it.value();
        }

        scored.push_back({doc, score});
    }

    // ============================================================
    // 6. Top-k selection
    // ============================================================

    if (scored.size() > static_cast<size_t>(top_k)) {

        std::nth_element(
            scored.begin(),
            scored.begin() + top_k,
            scored.end(),
            [](const Result& a, const Result& b) {
                return a.score > b.score;
            }
        );

        scored.resize(top_k);
    }

    // ============================================================
    // 7. Final descending sort
    // ============================================================

    std::sort(
        scored.begin(),
        scored.end(),
        [](const Result& a, const Result& b) {
            return a.score > b.score;
        }
    );

    return scored;
}