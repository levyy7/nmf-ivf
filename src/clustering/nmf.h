#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <functional>
#include <string>
#include <vector>

// =====================================================================
// NMFModel
// =====================================================================
//
// Decomposes X ≈ W H  (W ≥ 0, H ≥ 0) using one of two solvers:
//
//   HALS (default):
//     - W update: HALS (Cichocki 2007), parallel over k components (OMP)
//     - H update: HALS, parallel over k components (OMP + sparse CSR)
//     - Best for moderate-size matrices or when you need provable convergence.
//
//   SGD (Hogwild!):
//     - Joint W+H update over observed non-zeros only — never materialises
//       the full residual matrix.
//     - Parallelism: OMP parallel-for over documents. W[i,:] is exclusive
//       per thread; H[:,j] is shared without locks (Hogwild! benign race).
//     - Best for very large sparse matrices (e.g. SPLADE at ~99% sparsity).
//
// Only H is stored after fit(). W is a local variable during training and
// is discarded; use project() to re-derive W for new data:
//   W_approx = max(X @ H.T, 0)
//
// ── Supported input types ────────────────────────────────────────────
//   SparseMat = Eigen::SparseMatrix<float, RowMajor>   (SPLADE, recommended)
//   DenseMat  = Eigen::MatrixXf
//
// ── Usage ────────────────────────────────────────────────────────────
//   NMFModel::Config cfg;
//   cfg.k        = 256;
//   cfg.solver   = NMFModel::SolverType::SGD;
//   cfg.lr       = 0.005f;
//   NMFModel model(cfg);
//   model.fit(X_sparse);
//   Eigen::MatrixXf W_batch = model.project(X_batch);

class NMFModel {
public:
    // ── Types ────────────────────────────────────────────────────────
    using DenseMat  = Eigen::MatrixXf;
    using SparseMat = Eigen::SparseMatrix<float, Eigen::RowMajor>;
    using Vec       = Eigen::VectorXf;

    using ProgressCallback = std::function<bool(int iter, float error)>;

    // ── Solver and init strategies ───────────────────────────────────
    enum class SolverType { HALS, SGD };
    enum class InitType   { Random };

    // ── Configuration ────────────────────────────────────────────────
    struct Config {
        // Factorisation
        int k = 512;                        // number of latent components

        // Stopping
        int   max_iter    = 100;            // maximum iterations / epochs
        float tol         = 1e-4f;          // stop when rel H change < tol
        int   check_every = 5;              // convergence check interval

        // Initialisation
        InitType init  = InitType::Random;
        int      seed  = 42;
        float    scale = 1.f;               // set automatically by fit()

        // Solver
        SolverType solver = SolverType::HALS;

        // SGD hyperparameters (only used when solver == SGD)
        float lr       = 0.005f;            // initial learning rate
        float lr_decay = 0.95f;             // per-epoch LR multiplier  (0,1]
        float reg      = 1e-5f;             // L2 regularisation on W and H
        bool  shuffle  = true;              // shuffle doc order each epoch

        // Runtime
        int  n_threads = 0;                 // 0 = use Eigen/OMP defaults
        bool verbose   = true;
    };

    // ── Construction ─────────────────────────────────────────────────
    explicit NMFModel(const Config& cfg);

    // Convenience constructor — HALS solver, random init
    explicit NMFModel(int k, int max_iter = 100, float tol = 1e-4f);

    // ── fit() ────────────────────────────────────────────────────────
    // Learn H from X (n_samples × vocab). Accepts sparse or dense input.
    void fit(const SparseMat& X, ProgressCallback cb = nullptr);
    void fit(const DenseMat&  X, ProgressCallback cb = nullptr);

    // ── project() ────────────────────────────────────────────────────
    // Returns W_approx = max(X @ H.T, 0),  shape (n, k).
    // Parallelism: Eigen BLAS threads for the matmul.
    DenseMat project(const SparseMat& X) const;
    DenseMat project(const DenseMat&  X) const;

    // ── Accessors ────────────────────────────────────────────────────
    const DenseMat& components() const { return H_; }   // shape (k, vocab)
    int             k()          const { return cfg_.k; }
    float           last_error() const { return last_error_; }
    int             iters_run()  const { return iters_run_; }
    bool            is_fitted()  const { return fitted_; }

private:
    Config   cfg_;
    DenseMat H_;                            // (k, vocab) — kept after fit
    float    last_error_ = 0.f;
    int      iters_run_  = 0;
    bool     fitted_     = false;

    // ── Core fit loop (sparse and dense share one implementation) ────
    template <typename MatType>
    void fit_impl(const MatType& X, ProgressCallback cb);

    // ── HALS: W update ───────────────────────────────────────────────
    // Updates W in-place, one component at a time (leave-one-out trick).
    // HHt = H H^T and XHt = X H^T are precomputed by the caller.
    // Parallelism: OMP parallel-for over k components.
    template <typename MatType>
    void update_W_hals(DenseMat& W,
                       const DenseMat& HHt,
                       const DenseMat& XHt) const;

    // ── HALS: H update ───────────────────────────────────────────────
    // H ← max(0, H + (W^T X - W^T W H) / diag(W^T W))
    // Sparse path: raw CSR loop per component, OMP-parallel over k.
    // Dense path:  BLAS dgemv per component, OMP-parallel over k.
    template <typename MatType>
    void update_H_hals(const DenseMat& WtW,
                                  const DenseMat& WtX);

    // ── SGD: joint W+H Hogwild! update ───────────────────────────────
    // Iterates only over observed non-zeros; never materialises W H.
    // W[i,:] is thread-private; H[:,j] is shared (intentional benign race).
    // perm gives the document processing order (shuffled each epoch).
    template <typename MatType>
    void update_WH_sgd(const MatType& X, DenseMat& W,
                       const std::vector<int>& perm, float lr);

    // ── Convergence metrics ──────────────────────────────────────────
    // Relative Frobenius change in H between iterations.
    float relative_h_change(const DenseMat& H_old) const;

    // ||X - WH||_F using the trace trick — no full residual allocation.
    // Used by HALS convergence check.
    template <typename MatType>
    float reconstruction_error(const MatType& X,
                                      const DenseMat& W,
                                      const DenseMat& WtX,
                                      const DenseMat& WtW) const;

    // RMSE over observed non-zeros — O(nnz * k), cheap.
    // Used by SGD convergence check.
    template <typename MatType>
    float nnz_rmse(const MatType& X, const DenseMat& W) const;

    // ── Initialisation ───────────────────────────────────────────────
    void init_factors(int n, int vocab, DenseMat& W, DenseMat& H) const;
    void init_random (int n, int vocab, DenseMat& W, DenseMat& H) const;
};