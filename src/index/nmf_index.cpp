#include "nmf_index.h"

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <queue>
#include <random>

#include "nmf/base.h"


NMFIndex::NMFIndex(std::unique_ptr<NMFBase> nmf,
                        const Config& idx_cfg)
    : nmf_(std::move(nmf)),
      cfg_(idx_cfg),
      X_docs_(nullptr) {}


void NMFIndex::build(const SpMat& X) {
    auto t0 = std::chrono::high_resolution_clock::now();

    if (cfg_.verbose) {
        std::cout << "NMFIndex::build()\n";
    }
    X_docs_ = &X;
    n_docs_ = static_cast<int>(X.rows());

    //TODO: Data is duplicated to X_fit! Change to reference
    SpMat X_fit;
    if (X.rows() > cfg_.sample_size)
    {
        // Randomly sample rows without copying the entire matrix
        std::vector<int> indices(X.rows());
        std::iota(indices.begin(), indices.end(), 0);

        std::shuffle(indices.begin(), indices.end(),
                     std::mt19937{std::random_device{}()});

        indices.resize(cfg_.sample_size);

        // Create sampled matrix
        X_fit.resize(cfg_.sample_size, X.cols());

        for (int i = 0; i < cfg_.sample_size; ++i) {
            X_fit.row(i) = X.row(indices[i]);
        }
        compute_H(X_fit);
    }
    else {
        compute_H(X);
    }

    // 2. Build inverted lists
    build_lists(X);

    built_ = true;

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (cfg_.verbose) {
        std::cout << "Index built: n_docs=" << n_docs_
                  << " k=" << H_.rows()
                  << " | build time: " << ms << " ms\n";
    }
}



void NMFIndex::compute_H(const SpMat& X_fit)
{
    nmf_->fit(X_fit);
    H_ = nmf_->components();
}

void NMFIndex::build_lists(const SpMat& X) {
    std::cout << "NMFIndex::build_lists()\n";
    const int n = static_cast<int>(X.rows());
    const int k = static_cast<int>(H_.rows());
    const int m = cfg_.m;

    lists_.assign(k, {});

    #pragma omp parallel for schedule(dynamic)
    for (int topic = 0; topic < k; ++topic) {

        const Eigen::VectorXf h = H_.row(topic).transpose();

        // Min-heap capped at m entries — never grows beyond m
        // Entry: (score, doc_id)
        using Pair = std::pair<float, int>;
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> heap;
        for (int i = 0; i < n; ++i) {
            float score = X.row(i).dot(h);
            if (score <= 0.f) continue;

            if (static_cast<int>(heap.size()) < m) {
                heap.emplace(score, i);
            } else if (score > heap.top().first) {
                heap.pop();
                heap.emplace(score, i);
            }
        }

        // Drain heap into sorted list (descending)
        std::vector<ListEntry> tmp;
        tmp.resize(heap.size());
        int idx = static_cast<int>(heap.size()) - 1;
        while (!heap.empty()) {
            tmp[idx--] = {heap.top().second, heap.top().first};
            heap.pop();
        }

        lists_[topic] = std::move(tmp);
    }
}


int NMFIndex::list_size(int r) const {
    if (r < 0 || r >= static_cast<int>(lists_.size()))
        return 0;
    return static_cast<int>(lists_[r].size());
}


std::vector<std::vector<NMFIndex::Result>>
NMFIndex::search(const SpMat& queries, int top_k, int nprobe) const
{
    std::cout << "NMFIndex::search()\n";
    if (!built_)
        throw std::runtime_error("Index not built");

    if (nprobe < 0) nprobe = cfg_.nprobe;

    const int Q = static_cast<int>(queries.rows());
    std::vector<std::vector<Result>> results(Q);

    const int CHUNK = 64; // tune based on available RAM
    #pragma omp parallel for schedule(dynamic)
    for (int qi = 0; qi < Q; qi += CHUNK) {
        const int end = std::min(qi + CHUNK, Q);
        const int chunk_size = end - qi;

        // Project only this chunk
        Eigen::MatrixXf chunk_projected =
            queries.middleRows(qi, chunk_size) * H_.transpose(); // (CHUNK × k)

        for (int i = 0; i < chunk_size; ++i) {
            Eigen::SparseVector<float, Eigen::RowMajor> query = queries.row(qi + i);
            results[qi + i] = search_one(query, chunk_projected.row(i), top_k, nprobe);
        }
    }

    return results;
}


std::vector<NMFIndex::Result>
NMFIndex::search_one(
                    const Eigen::SparseVector<float, Eigen::RowMajor>& query,
                    const Eigen::RowVectorXf& query_scores,
                    int top_k,
                    int nprobe) const
{
    const int k = static_cast<int>(query_scores.size());

    // 2. Pick top-nprobe latent components
    std::vector<int> comps(k);
    std::iota(comps.begin(), comps.end(), 0);

    const int probes = std::min(nprobe, k);

    std::partial_sort(
        comps.begin(),
        comps.begin() + probes,
        comps.end(),
        [&](int a, int b) {
            return query_scores[a] > query_scores[b];
        }
    );

    // 3. Gather candidates from inverted lists
    std::vector<int> candidates;
    candidates.reserve(cfg_.m * probes);

    for (int i = 0; i < probes; ++i) {
        const int r = comps[i];

        for (const auto& e : lists_[r]) {
            candidates.push_back(e.doc_id);
        }
    }

    // 4. Deduplicate candidates
    std::sort(candidates.begin(), candidates.end());

    candidates.erase(
        std::unique(candidates.begin(), candidates.end()),
        candidates.end()
    );

    // 5. FINAL RERANK IN ORIGINAL SPACE
    //
    // score(q, d) = q · x_d
    //
    // q  : dense query vector
    // x_d: sparse document row
    std::vector<Result> scored;
    scored.reserve(candidates.size());

    for (int doc : candidates) {

        float score = 0.f;

        // Sparse dot product:
        // iterate only over non-zeros of document row
        for (SpMat::InnerIterator it(*X_docs_, doc); it; ++it) {
            score += query.coeff(it.col()) * it.value();
        }

        scored.push_back({doc, score});
    }

    // 6. Top-k selection
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

    // 7. Final descending sort
    std::sort(
        scored.begin(),
        scored.end(),
        [](const Result& a, const Result& b) {
            return a.score > b.score;
        }
    );

    return scored;
}