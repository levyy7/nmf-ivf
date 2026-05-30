#include "index/backend/base.h"

#include <iostream>
#include <queue>
#include <utility>
#include <vector>


void IVFBackend::build(const SpMat& X, Eigen::MatrixXf H) {
  H_ = std::move(H);

  const int n = static_cast<int>(X.rows());
  const int k = static_cast<int>(H_.rows());
  const int m = static_cast<int>(std::ceil(
      (static_cast<double>(n) / k) * overlap_factor_));

  if (cfg_.verbose) {
    std::cout << "[IVFBackend] Starting index build...\n";
    std::cout << "[IVFBackend] Documents (n): " << X.rows() << "\n";
    std::cout << "[IVFBackend] Topics (k):    " << H_.rows() << "\n";
    std::cout << "[IVFBackend] List cap (m):  " << m << "\n";
  }

  lists_ = Eigen::MatrixXi::Constant(k, m, -1);

#pragma omp parallel for schedule(dynamic)
  for (int topic = 0; topic < k; ++topic) {
    const Eigen::VectorXf h = H_.row(topic).transpose();

    // Min-heap capped at m entries
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

    // Drain heap into the Eigen matrix row (populated right-to-left for descending order)
    int num_elements = static_cast<int>(heap.size());
    int idx = num_elements - 1;
    while (!heap.empty()) {
      lists_(topic, idx) = heap.top().second;
      heap.pop();
      idx--;
    }
  }

  built_ = true;

  if (cfg_.verbose) {
    std::cout << "[IVFBackend] Build complete. Index is ready.\n";
  }
}

std::vector<std::vector<IVFBackend::SearchResult>> IVFBackend::search(
    const SpMat& queries, const SpMat& X_docs,
    int top_k, const SearchParams* params) const {
  if (!built_) {
    throw std::runtime_error("IVFBackend::search — index is not built");
  }

  const int Q = static_cast<int>(queries.rows());
  std::vector<std::vector<SearchResult>> results(Q);

  const int CHUNK = 64;

  // Parallelize query evaluation over chunks
#pragma omp parallel for schedule(dynamic)
  for (int qi = 0; qi < Q; qi += CHUNK) {
    const int end = std::min(qi + CHUNK, Q);
    const int chunk_size = end - qi;

    // Project the chunk of queries into the component space
    Eigen::MatrixXf chunk_projected =
        queries.middleRows(qi, chunk_size) * H_.transpose();

    // Evaluate each query in the chunk
    for (int i = 0; i < chunk_size; ++i) {
      Eigen::SparseVector<float, Eigen::RowMajor> query = queries.row(qi + i);
      results[qi + i] = search_one(query, chunk_projected.row(i), X_docs, top_k,
                                   params);
    }
  }

  return results;
}
