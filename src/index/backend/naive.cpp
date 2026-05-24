#include "naive.h"

#include <vector>
#include <queue>
#include <numeric>
#include <algorithm>
#include <cmath>

std::vector<IVFBackend::SearchResult> NaiveIVFBackend::search_one(
    const Eigen::SparseVector<float, Eigen::RowMajor>& query,
    const Eigen::RowVectorXf& query_scores,
    const SpMat& X_docs,
    int top_k,
    const IVFBackend::SearchParams* params) const
{
    const int k = static_cast<int>(query_scores.size());
    const int n_docs = static_cast<int>(X_docs.rows());

    int active_nprobe = default_nprobe;
    float active_factor = default_list_search_factor;

    if (params) {
        if (auto p = dynamic_cast<const SearchParams*>(params)) {
            active_nprobe = p->nprobe;
            active_factor = p->list_search_factor;
        }
    }

    // Sort list indices by projection score (descending)
    std::vector<int> list_order(k);
    std::iota(list_order.begin(), list_order.end(), 0);
    std::sort(list_order.begin(), list_order.end(), [&](int a, int b) {
        return query_scores[a] > query_scores[b];
    });

    // Set up a Min-Heap for Top-K
    auto cmp = [](const SearchResult& a, const SearchResult& b) {
        return a.score > b.score;
    };
    std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(cmp)> top_k_heap(cmp);
    std::vector<bool> visited(n_docs, false);

    int num_probes = std::min(active_nprobe, k);
    int max_docs_per_list = static_cast<int>(std::round(lists_.cols() * active_factor));

    max_docs_per_list = std::max(1, std::min(max_docs_per_list, static_cast<int>(lists_.cols())));

    for (int p = 0; p < num_probes; ++p) {
        int r = list_order[p];

        // Only iterate up to max_docs_per_list instead of the whole column
        for (int c = 0; c < max_docs_per_list; ++c) {
            int doc = lists_(r, c);

            if (doc < 0) break; // Reached padding, list has no more docs

            if (doc >= n_docs) continue;
            if (visited[doc]) continue;
            visited[doc] = true;

            float score = query.dot(X_docs.row(doc));

            if (static_cast<int>(top_k_heap.size()) < top_k) {
                top_k_heap.push({doc, score});
            } else if (score > top_k_heap.top().score) {
                top_k_heap.pop();
                top_k_heap.push({doc, score});
            }
        }
    }

    std::vector<SearchResult> results(top_k_heap.size());
    int idx = static_cast<int>(top_k_heap.size()) - 1;
    while (!top_k_heap.empty()) {
        results[idx--] = top_k_heap.top();
        top_k_heap.pop();
    }

    return results;
}