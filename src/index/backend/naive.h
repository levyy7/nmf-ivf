#pragma once

#include "base.h"

class NaiveIVFBackend : public IVFBackend {
public:
    // ── Configuration ─────────────────────────────────────────────────────
    // Inherits base config and adds `nprobe`
    struct SearchParams : IVFBackend::SearchParams {
        int nprobe = default_nprobe;
        float list_search_factor = default_list_search_factor;

        explicit SearchParams(int nprobe, float list_search_factor) :
            nprobe(nprobe),
            list_search_factor(list_search_factor) {}
    };

    // ── Constructors ──────────────────────────────────────────────────

    // 1. Unbuilt state
    explicit NaiveIVFBackend(const Config& cfg = Config())
        : IVFBackend(cfg) {}

    // 2. Built state
    NaiveIVFBackend(Eigen::MatrixXf H, Eigen::MatrixXi lists, const Config& cfg = Config())
        : IVFBackend(std::move(H), std::move(lists))
    {
        // Update the base class config manually since the base built-constructor
        // doesn't take a Config struct.
        this->cfg_ = cfg;
    }

protected:
    static constexpr int default_nprobe = 10;
    static constexpr float default_list_search_factor = 1.0;

    // ── Search Implementation ─────────────────────────────────────────
    [[nodiscard]] std::vector<SearchResult> search_one(
        const Eigen::SparseVector<float, Eigen::RowMajor>& query,
        const Eigen::RowVectorXf& query_scores,
        const SpMat& X_docs,
        int top_k,
        const IVFBackend::SearchParams* params) const override;
};