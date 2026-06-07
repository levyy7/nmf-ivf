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
  const int vocab_size = static_cast<int>(X_docs.cols()); // 30,522 for SPLADE

  thread_local std::vector<int> visited;
  thread_local int current_epoch = 0;
  thread_local std::vector<std::pair<float, int>> filtered_lists;

  thread_local std::vector<float> dense_query;

  if (visited.size() != static_cast<size_t>(n_docs)) {
    visited.assign(n_docs, 0);
    current_epoch = 0;
  }
  if (dense_query.size() != static_cast<size_t>(vocab_size)) {
    dense_query.assign(vocab_size, 0.0f);
  }

  current_epoch++;
  if (current_epoch <= 0) {
    visited.assign(n_docs, 0);
    current_epoch = 1;
  }

  // Parse parameters
  int max_misses = 30;
  float drop_ratio = 0.15f;
  if (params) {
    if (auto p = dynamic_cast<const SearchParams*>(params)) {
      max_misses = p->max_consecutive_misses;
      drop_ratio = p->score_drop_ratio;
    }
  }

  // Find the max cluster affinity to compute thresholds
  float max_cluster_score = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < k; ++i) {
    if (query_scores[i] > max_cluster_score) {
      max_cluster_score = query_scores[i];
    }
  }
  float cutoff_score = max_cluster_score * drop_ratio;

  filtered_lists.clear();
  for (int i = 0; i < k; ++i) {
    float q_score = query_scores[i];
    if (q_score > 0.0f && q_score >= cutoff_score) {
      filtered_lists.emplace_back(q_score, i);
    }
  }

  std::sort(filtered_lists.begin(), filtered_lists.end(),
            [](const auto& a, const auto& b) {
              return a.first > b.first;
            });

  for (Eigen::SparseVector<float, Eigen::RowMajor>::InnerIterator it(query); it;
       ++it) {
    dense_query[it.index()] = it.value();
  }

  auto cmp = [](const SearchResult& a, const SearchResult& b) {
    return a.score > b.score;
  };
  std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(cmp)>
      top_k_heap(cmp);

  float min_score_to_enter = -std::numeric_limits<float>::infinity();
  int max_cols = static_cast<int>(lists_.cols());

  // Search through clusters
  for (const auto& list_pair : filtered_lists) {
    int r = list_pair.second;
    int consecutive_misses = 0;

    for (int c = 0; c < max_cols; ++c) {
      int doc = lists_(r, c);

      if (doc < 0) break; // End of cluster padding
      if (doc >= n_docs) continue;

      if (visited[doc] == current_epoch) continue;
      visited[doc] = current_epoch;

      float score = 0.0f;
      for (SpMat::InnerIterator it(X_docs, doc); it; ++it) {
        score += it.value() * dense_query[it.index()];
      }

      bool made_top_k = false;
      if (static_cast<int>(top_k_heap.size()) < top_k) {
        top_k_heap.push({doc, score});
        made_top_k = true;
        if (static_cast<int>(top_k_heap.size()) == top_k) {
          min_score_to_enter = top_k_heap.top().score;
        }
      } else if (score > min_score_to_enter) {
        top_k_heap.pop();
        top_k_heap.push({doc, score});
        min_score_to_enter = top_k_heap.top().score;
        made_top_k = true;
      }

      if (made_top_k) {
        consecutive_misses = 0;
      } else {
        consecutive_misses++;
      }

      if (consecutive_misses >= max_misses) {
        break;
      }
    }
  }

  for (Eigen::SparseVector<float, Eigen::RowMajor>::InnerIterator it(query); it;
       ++it) {
    dense_query[it.index()] = 0.0f;
  }

  // Output formatting
  std::vector<SearchResult> results(top_k_heap.size());
  int idx = static_cast<int>(top_k_heap.size()) - 1;
  while (!top_k_heap.empty()) {
    results[idx--] = top_k_heap.top();
    top_k_heap.pop();
  }

  return results;
}