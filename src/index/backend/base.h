#pragma once

#include <vector>
#include <algorithm>

#include <Eigen/Sparse>
#include "utils/eigen_utils.h"

class IVFBackend {
public:
  struct Config {
    bool verbose;

    Config(bool v_val = false)
      : verbose(v_val) {
    }
  };

  struct SearchParams {
    virtual ~SearchParams() = default;
  };

  struct SearchResult {
    int id;
    float score;

    // Allows easy sorting (e.g., standard priority queues)
    bool operator<(const SearchResult& other) const {
      return score < other.score;
    }
  };

  virtual ~IVFBackend() = default;

  explicit IVFBackend(const Config& cfg = Config())
    : cfg_(cfg), built_(false) {
  }

  IVFBackend(Eigen::MatrixXf H, Eigen::MatrixXi lists)
    : H_(std::move(H)), lists_(std::move(lists)), built_(true) {
  }

  void build(const SpMat& X, Eigen::MatrixXf H);

  [[nodiscard]]
  std::vector<std::vector<SearchResult>> search(
      const SpMat& queries,
      const SpMat& X_docs,
      int top_k,
      const SearchParams* params = nullptr) const;


  bool is_built() const { return built_; }
  const Eigen::MatrixXf& components() const { return H_; }
  const Eigen::MatrixXi& lists() const { return lists_; }

  int n_lists() const {
    return static_cast<int>(lists_.rows());
  }

  int list_size(int r) const {
    if (r < 0 || r >= lists_.rows()) return 0;
    int count = 0;
    for (int c = 0; c < lists_.cols(); ++c) {
      if (lists_(r, c) >= 0) count++;
      else break; // padding -1 means the rest of the row is empty
    }
    return count;
  }

protected:
  Eigen::MatrixXf H_;
  Eigen::MatrixXi lists_; // k x m matrix containing document IDs
  Config cfg_;
  bool built_ = false;
  static constexpr float overlap_factor_ = 10.0f;

  [[nodiscard]] virtual std::vector<SearchResult> search_one(
      const Eigen::SparseVector<float, Eigen::RowMajor>& query,
      const Eigen::RowVectorXf& query_scores,
      const SpMat& X_docs,
      int top_k,
      const SearchParams* params) const = 0;
};