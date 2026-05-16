#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE
#include "nmf.h"

#include <cblas.h>
#include <iostream>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>


// =====================================================================
// Construction
// =====================================================================

NMFModel::NMFModel(const Config& cfg) : cfg_(cfg) {}

NMFModel::NMFModel(int k, int max_iter, float tol) {
    cfg_.k        = k;
    cfg_.max_iter = max_iter;
    cfg_.tol      = tol;
}

// =====================================================================
// Public fit()
// =====================================================================

void NMFModel::fit(const SparseMat& X, ProgressCallback cb) {
    if (cfg_.verbose) {
        std::cout << "OpenBLAS threads: " << openblas_get_num_threads() << "\n";
#ifdef EIGEN_USE_BLAS
        std::cout << "EIGEN_USE_BLAS: defined\n";
#else
        std::cout << "EIGEN_USE_BLAS: NOT defined — Eigen using own impl\n";
#endif
    }
    fit_impl(X, std::move(cb));
}

void NMFModel::fit(const DenseMat& X, ProgressCallback cb) {
    fit_impl(X, std::move(cb));
}

// =====================================================================
// Public project()
// =====================================================================

NMFModel::DenseMat NMFModel::project(const SparseMat& X) const {
    if (!fitted_)
        throw std::runtime_error("NMFModel: call fit() before project().");
    return (X * H_.transpose()).cwiseMax(0.f);
}

NMFModel::DenseMat NMFModel::project(const DenseMat& X) const {
    if (!fitted_)
        throw std::runtime_error("NMFModel: call fit() before project().");
    return (X * H_.transpose()).cwiseMax(0.f);
}

// =====================================================================
// Private: fit_impl
// =====================================================================

template <typename MatType>
void NMFModel::fit_impl(const MatType& X, ProgressCallback cb) {
    const int n     = static_cast<int>(X.rows());
    const int vocab = static_cast<int>(X.cols());

    if (n == 0 || vocab == 0)
        throw std::invalid_argument("NMFModel: empty input matrix.");
    if (cfg_.k <= 0 || cfg_.k > std::min(n, vocab))
        throw std::invalid_argument(
            "NMFModel: k must be in [1, min(n_samples, vocab)].");

    if (cfg_.n_threads > 0) {
        Eigen::setNbThreads(cfg_.n_threads);
        omp_set_num_threads(cfg_.n_threads);
    }

    // ── Initialise ──────────────────────────────────────────────────
    float x_mean;
    if constexpr (std::is_same_v<MatType, SparseMat>)
        x_mean = X.sum() / static_cast<float>(X.rows() * X.cols());
    else
        x_mean = X.mean();

    cfg_.scale = 2.f * std::sqrt(x_mean / static_cast<float>(cfg_.k));

    DenseMat W, H;
    init_factors(n, vocab, W, H);
    H_ = std::move(H);

    if (cfg_.verbose) {
        const char* solver_name =
            cfg_.solver == SolverType::SGD ? "SGD (Hogwild!)" : "HALS";
        std::cout << "NMF fit\n"
                  << "  solver="      << solver_name
                  << "  n="           << n
                  << "  vocab="       << vocab
                  << "  k="           << cfg_.k
                  << "  max_iter="    << cfg_.max_iter << "\n"
                  << "  x_mean="      << x_mean
                  << "  init_scale="  << cfg_.scale    << "\n"
                  << "  threads(blas)=" << Eigen::nbThreads()
                  << "  threads(omp)="  << omp_get_max_threads() << "\n";
        if (cfg_.solver == SolverType::SGD)
            std::cout << "  lr="       << cfg_.lr
                      << "  lr_decay=" << cfg_.lr_decay
                      << "  reg="      << cfg_.reg << "\n";
        else
            std::cout << "  reg=" << cfg_.reg << "\n";
    }

    // ── SGD path ────────────────────────────────────────────────────
    if (cfg_.solver == SolverType::SGD) {

        std::mt19937 rng(static_cast<unsigned>(cfg_.seed) + 1337u);
        std::vector<int> perm(n);
        std::iota(perm.begin(), perm.end(), 0);

        float lr = cfg_.lr;

        for (int iter = 0; iter < cfg_.max_iter; ++iter) {
            const DenseMat H_old = H_;

            if (cfg_.shuffle)
                std::shuffle(perm.begin(), perm.end(), rng);

            update_WH_sgd(X, W, perm, lr);
            lr *= cfg_.lr_decay;
            ++iters_run_;

            if (iter % cfg_.check_every == 0 || iter == cfg_.max_iter - 1) {
                const float rel_change = relative_h_change(H_old);
                last_error_            = nnz_rmse(X, W);

                if (cfg_.verbose)
                    std::cout << "  iter "         << iter
                              << "  RMSE_nnz="     << last_error_
                              << "  rel_H_change=" << rel_change
                              << "  lr="           << lr << "\n";

                if (cb && !cb(iter, last_error_)) {
                    if (cfg_.verbose) std::cout << "  Early stop via callback.\n";
                    break;
                }
                if (rel_change < cfg_.tol) {
                    if (cfg_.verbose)
                        std::cout << "  Converged (rel_change=" << rel_change
                                  << " < tol=" << cfg_.tol << ").\n";
                    break;
                }
            }
        }

        fitted_ = true;
        return;
    }

    // ── HALS path ───────────────────────────────────────────────────
    //
    // We precompute the Gram matrices once per iteration and reuse them
    // across both updates and the convergence check — no redundant recomputes.
    //
    // Iteration structure:
    //   1. Compute HHt and XHt  (BLAS multi-threaded)
    //   2. HALS update W        (OMP parallel over components)
    //   3. Normalise W columns and absorb scale into H rows
    //   4. Compute WtW and WtX  (BLAS multi-threaded)
    //   5. HALS update H        (OMP parallel over components)
    //   6. Convergence check using the already-computed Gram matrices

    for (int iter = 0; iter < cfg_.max_iter; ++iter) {
        const DenseMat H_old = H_;

        // ── Step 1: Gram matrices for the W update ───────────────────
        // Give all cores to BLAS for the dense matrix products.
        openblas_set_num_threads(omp_get_max_threads());
        Eigen::setNbThreads(omp_get_max_threads());

        DenseMat HHt(cfg_.k, cfg_.k);
        HHt.setZero();
        HHt.selfadjointView<Eigen::Lower>().rankUpdate(H_);
        HHt = HHt.selfadjointView<Eigen::Lower>();   // dsyrk

        // XHt = X * H^T,  shape (n, k)
        // Sparse path uses Eigen's sparse-dense product (CSR × dense).
        const DenseMat XHt = X * H_.transpose();

        // ── Step 2: HALS W update ────────────────────────────────────
        // Disable BLAS threading before entering the OMP region to avoid
        // nested parallelism (BLAS inside OMP threads).
        openblas_set_num_threads(1);
        Eigen::setNbThreads(1);

        update_W_hals<MatType>(W, HHt, XHt);

        // ── Step 3: Column-normalise W, absorb norms into H ──────────
        // Done AFTER the full W update so every component sees a
        // consistent scale.  This keeps WH invariant while preventing
        // W columns from growing unboundedly.
        openblas_set_num_threads(omp_get_max_threads());
        Eigen::setNbThreads(omp_get_max_threads());

        for (int r = 0; r < cfg_.k; ++r) {
            const float norm = W.col(r).norm();
            if (norm > 1e-10f) {
                W.col(r)  /= norm;
                H_.row(r) *= norm;
            }
        }

        // ── Step 4: Gram matrices for the H update ───────────────────
        DenseMat WtW(cfg_.k, cfg_.k);
        WtW.setZero();
        WtW.selfadjointView<Eigen::Lower>().rankUpdate(W.transpose());
        WtW = WtW.selfadjointView<Eigen::Lower>();   // dsyrk

        // WtX = W^T * X, shape (k, vocab).
        // For sparse X we compute this once here and pass it in,
        // avoiding the per-component O(n × nnz_per_doc) accumulation
        // that the old code did inside the OMP loop.
        //
        // Sparse:  W^T (n×k)^T times CSR X  →  use X^T * W  then transpose.
        //          Eigen evaluates (X^T * W) as a sparse-dense product
        //          and the result is (vocab × k); we transpose to (k × vocab).
        // Dense:   straightforward BLAS dgemm.
        DenseMat WtX(cfg_.k, vocab);
        if constexpr (std::is_same_v<MatType, SparseMat>)
            WtX = (X.transpose() * W).transpose(); // (vocab,k)^T = (k,vocab)
        else
            WtX = W.transpose() * X;               // (k,vocab)

        // ── Step 5: HALS H update ────────────────────────────────────
        openblas_set_num_threads(1);
        Eigen::setNbThreads(1);

        update_H_hals<MatType>(WtW, WtX);

        ++iters_run_;

        // ── Step 6: Convergence check ────────────────────────────────
        if (iter % cfg_.check_every == 0 || iter == cfg_.max_iter - 1) {
            const float rel_change = relative_h_change(H_old);

            // Reuse WtW and HHt (recompute HHt for the new H_).
            // reconstruction_error() now accepts the precomputed WtX so
            // we don't redo the expensive sparse product.
            openblas_set_num_threads(omp_get_max_threads());
            Eigen::setNbThreads(omp_get_max_threads());
            last_error_ = reconstruction_error(X, W, WtX, WtW);
            openblas_set_num_threads(1);
            Eigen::setNbThreads(1);

            if (cfg_.verbose)
                std::cout << "  iter "          << iter
                          << "  ||X-WH||_F="    << last_error_
                          << "  rel_H_change="  << rel_change << "\n";

            if (cb && !cb(iter, last_error_)) {
                if (cfg_.verbose) std::cout << "  Early stop via callback.\n";
                break;
            }
            if (rel_change < cfg_.tol) {
                if (cfg_.verbose)
                    std::cout << "  Converged (rel_change=" << rel_change
                              << " < tol=" << cfg_.tol << ").\n";
                break;
            }
        }
    }

    fitted_ = true;
}

// =====================================================================
// Private: update_WH_sgd  (unchanged)
// =====================================================================

template <typename MatType>
void NMFModel::update_WH_sgd(const MatType& X, DenseMat& W,
                               const std::vector<int>& perm, float lr) {
    const int   n   = static_cast<int>(X.rows());
    const float reg = cfg_.reg;

    if constexpr (std::is_same_v<MatType, SparseMat>) {

        #pragma omp parallel for schedule(dynamic, 32)
        for (int pi = 0; pi < n; ++pi) {
            const int i = perm[pi];
            Eigen::RowVectorXf w_i = W.row(i);

            for (SparseMat::InnerIterator it(X, i); it; ++it) {
                const int   j = it.index();
                const float v = it.value();

                const Eigen::VectorXf h_j = H_.col(j);
                const float residual = v - w_i.dot(h_j);

                w_i += lr * (residual * h_j.transpose() - reg * w_i);
                w_i  = w_i.cwiseMax(0.f);

                H_.col(j) += lr * (residual * w_i.transpose() - reg * H_.col(j));
                H_.col(j)  = H_.col(j).cwiseMax(0.f);
            }

            W.row(i) = w_i;
        }

    } else {
        #pragma omp parallel for schedule(static)
        for (int pi = 0; pi < n; ++pi) {
            const int i = perm[pi];
            Eigen::RowVectorXf w_i = W.row(i);

            for (int j = 0; j < static_cast<int>(X.cols()); ++j) {
                const float v = X(i, j);
                if (v == 0.f) continue;

                const Eigen::VectorXf h_j    = H_.col(j);
                const float           residual = v - w_i.dot(h_j);

                w_i  += lr * (residual * h_j.transpose() - reg * w_i);
                w_i   = w_i.cwiseMax(0.f);
                H_.col(j) += lr * (residual * w_i.transpose() - reg * H_.col(j));
                H_.col(j)  = H_.col(j).cwiseMax(0.f);
            }

            W.row(i) = w_i;
        }
    }
}

// =====================================================================
// Private: update_W_hals
//
// True sequential HALS update for W (n x k), given precomputed
// HHt (k x k) and XHt (n x k).
//
// Same Gauss-Seidel rank-1 maintenance as update_H_hals:
//
//   Before the loop:   G = W * HHt            (n x k, full dgemm)
//   For each column r:
//     w_old     = W[:,r]
//     W[:,r]    = max(0,  (XHt[:,r] - G[:,r] + w_old * HHt[r,r])
//                         / (HHt[r,r] + reg + eps) )
//     delta     = W[:,r] - w_old
//     G        += outer(delta, HHt[r,:])       (rank-1 update, O(n*k))
//
// Parallelism: rank-1 update parallelised over rows (n = 57 638 here).
// =====================================================================

template <typename MatType>
void NMFModel::update_W_hals(DenseMat& W,
                              const DenseMat& HHt,
                              const DenseMat& XHt) const {
    const float eps = 1e-10f;
    const float reg = cfg_.reg;

    // G = W * HHt,  (n x k) — full dgemm, multi-threaded BLAS.
    openblas_set_num_threads(omp_get_max_threads());
    Eigen::setNbThreads(omp_get_max_threads());
    DenseMat G = W * HHt;
    openblas_set_num_threads(1);
    Eigen::setNbThreads(1);

    for (int r = 0; r < cfg_.k; ++r) {
        const float denom = HHt(r, r) + reg + eps;

        const Eigen::VectorXf w_old = W.col(r);
        W.col(r) = ((XHt.col(r) - G.col(r) + w_old * HHt(r, r))
                    / denom).cwiseMax(0.f);

        const Eigen::VectorXf delta = W.col(r) - w_old;
        if (delta.squaredNorm() < 1e-20f) continue;

        const Eigen::RowVectorXf hht_row = HHt.row(r);   // (k,) read-only
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(G.rows()); ++i)
            G.row(i) += delta(i) * hht_row;
    }
}

// =====================================================================
// Private: update_H_hals
//
// True sequential HALS update for H (k x vocab).
//
// Standard HALS (Cichocki 2007) is a *sequential* (Gauss-Seidel) method:
// when updating row r, rows 0..r-1 have already been replaced with their
// new values.  Computing WtWH = WtW*H once upfront and reusing it for
// every row is wrong — every row sees the gradient from the *old* H,
// which causes near-zero rel_H_change after the first iteration.
//
// We fix this with a rank-1 maintenance strategy:
//
//   Before the loop:   G = WtW * H            (k x vocab, full dgemm)
//   For each row r:
//     h_old     = H[r,:]
//     H[r,:]    = max(0,  (WtX[r,:] - G[r,:] + h_old * WtW[r,r])
//                         / (WtW[r,r] + reg + eps) )
//     delta     = H[r,:] - h_old
//     G        += outer(WtW[:,r], delta)       (rank-1 update, O(k*vocab))
//
// After row r is updated, G reflects the current H so row r+1 sees
// the correct (partially updated) gradient.  Cost is O(k^2 * vocab)
// total — same asymptotic as the naive approach, but correct.
//
// Parallelism: the sequential Gauss-Seidel dependency prevents OMP
// over r.  The rank-1 outer-product add is parallelised over vocab
// columns instead, which is the large dimension here (30 522).
// =====================================================================

template <typename MatType>
void NMFModel::update_H_hals(const DenseMat& WtW,
                              const DenseMat& WtX) {
    const float eps = 1e-10f;
    const float reg = cfg_.reg;

    // G = WtW * H,  (k x vocab) — full dgemm, multi-threaded BLAS.
    openblas_set_num_threads(omp_get_max_threads());
    Eigen::setNbThreads(omp_get_max_threads());
    DenseMat G = WtW * H_;
    openblas_set_num_threads(1);
    Eigen::setNbThreads(1);

    for (int r = 0; r < cfg_.k; ++r) {
        const float denom = WtW(r, r) + reg + eps;

        // Save old row, compute new row (direct closed-form assignment).
        const Eigen::RowVectorXf h_old = H_.row(r);
        H_.row(r) = ((WtX.row(r) - G.row(r) + h_old * WtW(r, r))
                     / denom).cwiseMax(0.f);

        // Rank-1 update to keep G consistent with the new H[r,:].
        // G += WtW[:,r]  (outer)  delta,  where delta = H_new[r,:] - h_old.
        // Skip if the row didn't change (saves O(k*vocab) work).
        const Eigen::RowVectorXf delta = H_.row(r) - h_old;
        if (delta.squaredNorm() < 1e-20f) continue;

        const Eigen::VectorXf wtw_col = WtW.col(r);   // (k,) read-only
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < static_cast<int>(G.cols()); ++j)
            G.col(j) += wtw_col * delta(j);
    }
}

// =====================================================================
// Private: nnz_rmse  (unchanged)
// =====================================================================

template <typename MatType>
float NMFModel::nnz_rmse(const MatType& X, const DenseMat& W) const {
    double sum_sq = 0.0;
    long   count  = 0;

    if constexpr (std::is_same_v<MatType, SparseMat>) {
        #pragma omp parallel for reduction(+:sum_sq, count) schedule(dynamic, 64)
        for (int i = 0; i < static_cast<int>(X.rows()); ++i) {
            for (SparseMat::InnerIterator it(X, i); it; ++it) {
                const float pred = W.row(i).dot(H_.col(it.index()));
                const float r    = it.value() - pred;
                sum_sq += static_cast<double>(r * r);
                ++count;
            }
        }
    } else {
        #pragma omp parallel for reduction(+:sum_sq, count) schedule(static)
        for (int i = 0; i < static_cast<int>(X.rows()); ++i)
            for (int j = 0; j < static_cast<int>(X.cols()); ++j) {
                const float pred = W.row(i).dot(H_.col(j));
                const float r    = X(i, j) - pred;
                sum_sq += static_cast<double>(r * r);
                ++count;
            }
    }

    return count > 0 ? static_cast<float>(std::sqrt(sum_sq / count)) : 0.f;
}

// =====================================================================
// Private: relative_h_change  (unchanged)
// =====================================================================

float NMFModel::relative_h_change(const DenseMat& H_old) const {
    return (H_ - H_old).norm() / (H_old.norm() + 1e-10f);
}

// =====================================================================
// Private: reconstruction_error
//
// ||X - WH||²_F = ||X||²_F  -  2 <WtX, H>_F  +  tr(WtW · HHt)
//
// Now accepts precomputed WtX and WtW to avoid redundant products.
// HHt is recomputed here for the updated H_ (cheap k×k product).
// =====================================================================

template <typename MatType>
float NMFModel::reconstruction_error(const MatType& X,
                                      const DenseMat& W,
                                      const DenseMat& WtX,
                                      const DenseMat& WtW) const {
    float X_sq;
    if constexpr (std::is_same_v<MatType, SparseMat>)
        X_sq = X.cwiseProduct(X).sum();
    else
        X_sq = X.squaredNorm();

    // <WtX, H>_F  — element-wise dot of two (k×vocab) matrices
    const float cross = WtX.cwiseProduct(H_).sum();

    // HHt for the current H_
    DenseMat HHt(cfg_.k, cfg_.k);
    HHt.setZero();
    HHt.selfadjointView<Eigen::Lower>().rankUpdate(H_);
    HHt = HHt.selfadjointView<Eigen::Lower>();

    const float WH_sq = (WtW * HHt).trace();

    return std::sqrt(std::max(0.f, X_sq - 2.f * cross + WH_sq));
}

// =====================================================================
// Private: init_factors / init_random  (unchanged)
// =====================================================================

void NMFModel::init_factors(int n, int vocab, DenseMat& W, DenseMat& H) const {
    switch (cfg_.init) {
        case InitType::Random:
            init_random(n, vocab, W, H);
            return;
        default:
            throw std::invalid_argument("NMFModel: requested InitType not implemented.");
    }
}

void NMFModel::init_random(int n, int vocab, DenseMat& W, DenseMat& H) const {
    std::mt19937 rng(static_cast<unsigned>(cfg_.seed));
    std::uniform_real_distribution<float> dist(0.f, cfg_.scale);

    W.resize(n, cfg_.k);
    H.resize(cfg_.k, vocab);
    for (int i = 0; i < W.size(); ++i) W(i) = dist(rng);
    for (int i = 0; i < H.size(); ++i) H(i) = dist(rng);
}

// =====================================================================
// Explicit template instantiations
// =====================================================================

template void NMFModel::fit_impl<NMFModel::SparseMat>(
    const SparseMat&, ProgressCallback);
template void NMFModel::fit_impl<NMFModel::DenseMat>(
    const DenseMat&, ProgressCallback);

template void NMFModel::update_W_hals<NMFModel::SparseMat>(
    DenseMat&, const DenseMat&, const DenseMat&) const;
template void NMFModel::update_W_hals<NMFModel::DenseMat>(
    DenseMat&, const DenseMat&, const DenseMat&) const;

// update_H_hals no longer templates on MatType — it takes precomputed
// (WtW, WtX) dense matrices regardless of whether X was sparse or dense.
// A single explicit instantiation is not needed; the function is not a
// template.  The two lines below are intentionally removed.

template void NMFModel::update_WH_sgd<NMFModel::SparseMat>(
    const SparseMat&, DenseMat&, const std::vector<int>&, float);
template void NMFModel::update_WH_sgd<NMFModel::DenseMat>(
    const DenseMat&, DenseMat&, const std::vector<int>&, float);

template float NMFModel::reconstruction_error<NMFModel::SparseMat>(
    const SparseMat&, const DenseMat&, const DenseMat&, const DenseMat&) const;
template float NMFModel::reconstruction_error<NMFModel::DenseMat>(
    const DenseMat&, const DenseMat&, const DenseMat&, const DenseMat&) const;

template float NMFModel::nnz_rmse<NMFModel::SparseMat>(
    const SparseMat&, const DenseMat&) const;
template float NMFModel::nnz_rmse<NMFModel::DenseMat>(
    const DenseMat&, const DenseMat&) const;