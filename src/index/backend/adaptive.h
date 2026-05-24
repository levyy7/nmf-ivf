#pragma once

#include "base.h"

class AdaptiveIVFBackend : public IVFBackend {
public:
    struct SearchParams : IVFBackend::SearchParams {
        // --- Dynamic Early Stopping Heuristics ---
        int max_consecutive_misses = 30; // Stop list if X docs fail to make top-K
        float score_drop_ratio = 0.15f;  // Stop checking lists if query_score < best_score * ratio

        explicit SearchParams(int max_misses = 30, 
                              float drop_ratio = 0.15f) :
            max_consecutive_misses(max_misses),
            score_drop_ratio(drop_ratio) {}
    };

    explicit AdaptiveIVFBackend(const Config& cfg = Config())
        : IVFBackend(cfg) {}

    AdaptiveIVFBackend(Eigen::MatrixXf H, Eigen::MatrixXi lists, const Config& cfg = Config())
        : IVFBackend(std::move(H), std::move(lists))
    {
        this->cfg_ = cfg;
    }

protected:
    [[nodiscard]] std::vector<SearchResult> search_one(
        const Eigen::SparseVector<float, Eigen::RowMajor>& query,
        const Eigen::RowVectorXf& query_scores,
        const SpMat& X_docs,
        int top_k,
        const IVFBackend::SearchParams* params) const override;
};