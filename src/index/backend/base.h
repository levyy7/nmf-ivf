#pragma once

#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm> // for std::min

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "utils/eigen_utils.h"

class IVFBackend {
public:
    // ── Configuration ─────────────────────────────────────────────────────
    struct Config {
        int  m;
        bool verbose;

        Config(int m_val = 100, bool v_val = false)
            : m(m_val), verbose(v_val) {}
    };

    struct SearchParams {
        virtual ~SearchParams() = default;
    };

    // ── Search Result Structure ───────────────────────────────────────────
    struct SearchResult {
        int id;
        float score;

        // Allows easy sorting (e.g., standard priority queues)
        bool operator<(const SearchResult& other) const {
            return score < other.score;
        }
    };

    virtual ~IVFBackend() = default;

    // ── Constructors ──────────────────────────────────────────────────

    // 1. Unbuilt state: H_ is populated later during build()
    explicit IVFBackend(const Config& cfg = Config())
        : cfg_(cfg), built_(false) {}

    // 2. Built state: Loaded from disk.
    // Passed by value and moved to avoid unnecessary memory allocations.
    IVFBackend(Eigen::MatrixXf H, Eigen::MatrixXi lists)
        : H_(std::move(H)), lists_(std::move(lists)), built_(true) {}

    // ── Build ─────────────────────────────────────────────────────────
    // Passed by value + moved into H_ to avoid copies where possible.
    void build(const SpMat& X, Eigen::MatrixXf H);

    // ── Search ────────────────────────────────────────────────────────
    [[nodiscard]]
    std::vector<std::vector<SearchResult>> search(
            const SpMat& queries,
            const SpMat& X_docs,
            int          top_k,
            const SearchParams* params = nullptr) const;


    // ── Accessors ─────────────────────────────────────────────────────
    bool                    is_built()   const { return built_; }
    const Eigen::MatrixXf&  components() const { return H_; }
    const Eigen::MatrixXi&  lists()      const { return lists_; }

    int n_lists() const {
        return static_cast<int>(lists_.rows());
    }

    // Returns the number of valid (non-negative) document IDs in a specific list
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
    Config          cfg_;
    bool            built_ = false;

    // ── Pure-virtual implementation hooks ─────────────────────────────
    // Evaluates a single query. `query_scores` contains the projection of
    // the query onto the components matrix (H_).
    [[nodiscard]] virtual std::vector<SearchResult> search_one(
        const Eigen::SparseVector<float, Eigen::RowMajor>& query,
        const Eigen::RowVectorXf& query_scores,
        const SpMat& X_docs,
        int top_k,
        const SearchParams* params) const = 0;
};