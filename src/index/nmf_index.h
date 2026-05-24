#pragma once

#include <vector>
#include <memory>
#include <string>
#include <functional>

#include "index/nmf/base.h"
#include "index/backend/base.h"
#include "utils/eigen_utils.h"

// =====================================================================
// NMFIndex — Orchestrator for NMF Topic Modeling and Inverted Indexing
// =====================================================================
class NMFIndex {
public:
    // ── Configuration ────────────────────────────────────────────────
    struct Config {
        int  sample_size;
        bool verbose;

        // Explicit constructor prevents Clangd default-member errors
        Config(int s_val = 100000, bool v_val = true)
            : sample_size(s_val), verbose(v_val) {}
    };

    // ── Construction ─────────────────────────────────────────────────
    // A single constructor handles both built and unbuilt states.
    // The `backend` instance intrinsically tracks whether it is built.
    explicit NMFIndex(std::unique_ptr<IVFBackend> backend,
                      const Config& cfg = Config());

    // ── File I/O ─────────────────────────────────────────────────────

    // Factory signature to instantiate your specific derived IVFBackend.
    using BackendFactory = std::function<std::unique_ptr<IVFBackend>(const Eigen::MatrixXf&, const Eigen::MatrixXi&)>;

    // Loads H and lists from HDF5, uses the factory to build the backend,
    // and returns a ready-to-use NMFIndex.
    static std::unique_ptr<NMFIndex> load(const std::string& path,
                                          BackendFactory backend_factory,
                                          const Config& cfg = Config());

    // Saves the backend's H and lists matrices to an HDF5 file.
    void save_index(const std::string& path) const;

    // ── Build ────────────────────────────────────────────────────────
    // Takes ownership of `nmf` directly. Fits the model, extracts H,
    // delegates list creation to the backend, and then `nmf` is destroyed
    // as it goes out of scope.
    void build(const SpMat& X, const std::unique_ptr<NMFBase>& nmf);

    // ── Search ───────────────────────────────────────────────────────
    [[nodiscard]] std::vector<std::vector<IVFBackend::SearchResult>> search(
        const SpMat& queries,
        const SpMat& X_docs,
        int          top_k,
        const IVFBackend::SearchParams* params = nullptr) const;

    std::vector<std::vector<IVFBackend::SearchResult>> search_and_save(
        const std::string& path,
        const SpMat& queries,
        const SpMat& X_docs,
        int top_k,
        const IVFBackend::SearchParams* params = nullptr) const;

    // ── Accessors ────────────────────────────────────────────────────
    bool is_built() const;

    const IVFBackend& backend() const { return *backend_; }

    // Forwarded convenience accessors
    const Eigen::MatrixXf& components() const { return backend_->components(); }
    const Eigen::MatrixXi& lists()      const { return backend_->lists(); }

private:
    std::unique_ptr<IVFBackend> backend_;
    Config                      cfg_;
    float build_time_sec_;

    void save_results(const std::string& path,
                      const std::vector<std::vector<IVFBackend::SearchResult>>& results,
                      float query_time_sec) const;
};