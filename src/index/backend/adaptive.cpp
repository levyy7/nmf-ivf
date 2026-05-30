#include "adaptive.h"

#include <vector>
#include <queue>
#include <numeric>
#include <algorithm>
#include <iostream>

std::vector<IVFBackend::SearchResult> AdaptiveIVFBackend::search_one(
    const Eigen::SparseVector<float, Eigen::RowMajor>& query,
    const Eigen::RowVectorXf& query_scores,
    const SpMat& X_docs,
    int top_k,
    const IVFBackend::SearchParams* params) const {
  const int k = static_cast<int>(query_scores.size());
  const int n_docs = static_cast<int>(X_docs.rows());

  // Default parameters
  int max_misses = 30;
  float drop_ratio = 0.15f;

  if (params) {
    if (auto p = dynamic_cast<const SearchParams*>(params)) {
      max_misses = p->max_consecutive_misses;
      drop_ratio = p->score_drop_ratio;
    }
  }

  // Sort list indices by projection score (descending)
  std::vector<int> list_order(k);
  std::iota(list_order.begin(), list_order.end(), 0);
  std::sort(list_order.begin(), list_order.end(), [&](int a, int b) {
    return query_scores[a] > query_scores[b];
  });

  auto cmp = [](const SearchResult& a, const SearchResult& b) {
    return a.score > b.score;
  };
  std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(cmp)>
      top_k_heap(cmp);
  std::vector<bool> visited(n_docs, false);

  // Track the absolute best score to apply the drop_ratio heuristic
  float best_query_score = query_scores[list_order[0]];
  float cutoff_score = best_query_score * drop_ratio;

  int max_cols = static_cast<int>(lists_.cols());

  // Tracker for total documents fully evaluated
  int docs_processed = 0;

  for (int r : list_order) {
    float q_score = query_scores[r];

    // --- HEURISTIC 1: Inter-list Early Stopping ---
    // If this list's affinity is drastically lower than the top list,
    // or goes negative, we stop exploring lists entirely.
    if (q_score <= 0.0f || q_score < cutoff_score) {
      break;
    }

    int consecutive_misses = 0;

    for (int c = 0; c < max_cols; ++c) {
      int doc = lists_(r, c);

      if (doc < 0) break; // Padding reached, list is exhausted
      if (doc >= n_docs) continue;

      if (visited[doc]) continue;
      visited[doc] = true;

      // Increment processed counter right before the expensive dot product
      docs_processed++;

      float score = query.dot(X_docs.row(doc));

      bool made_top_k = false;
      if (static_cast<int>(top_k_heap.size()) < top_k) {
        top_k_heap.push({doc, score});
        made_top_k = true;
      } else if (score > top_k_heap.top().score) {
        top_k_heap.pop();
        top_k_heap.push({doc, score});
        made_top_k = true;
      }

      // --- HEURISTIC 2: Intra-list Early Stopping ---
      if (made_top_k) {
        consecutive_misses = 0; // Reset on success
      } else {
        consecutive_misses++;
      }

      if (consecutive_misses >= max_misses) {
        break; // List exhausted its promising candidates, jump to next list
      }
    }
  }

  std::vector<SearchResult> results(top_k_heap.size());
  int idx = static_cast<int>(top_k_heap.size()) - 1;
  while (!top_k_heap.empty()) {
    results[idx--] = top_k_heap.top();
    top_k_heap.pop();
  }

  // Output the metrics if verbose is true
  if (cfg_.verbose) {
    std::cout << "[AdaptiveIVFBackend] Documents processed for this query: "
        << docs_processed << std::endl;
  }

  return results;
}