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
// Public fit() / project()
// =====================================================================

void NMFModel::fit(const SparseMat& X, ProgressCallback cb) {
    if (cfg_.verbose) {
        std::cout << "OpenBLAS threads: " << openblas_get_num_threads() << "\n";
#ifdef EIGEN_USE_BLAS
        std::cout << "EIGEN_USE_BLAS: defined\n";
#else
        std::cout << "EIGEN_USE_BLAS: NOT defined\n";
#endif
    }
    fit_impl(X, std::move(cb));
}

void NMFModel::fit(const DenseMat& X, ProgressCallback cb) {
    fit_impl(X, std::move(cb));
}

NMFModel::DenseMat NMFModel::project(const SparseMat& X) const {
    if (!fitted_) throw std::runtime_error("NMFModel: call fit() before project().");
    return (X * H_.transpose()).cwiseMax(0.f);
}

NMFModel::DenseMat NMFModel::project(const DenseMat& X) const {
    if (!fitted_) throw std::runtime_error("NMFModel: call fit() before project().");
    return (X * H_.transpose()).cwiseMax(0.f);
}

// =====================================================================
// Threading helpers
//
// Rule: BLAS products always run with all T threads.
//       OMP parallel regions always run with all T threads.
//       Never both at the same time (no nested parallelism).
//
// So the pattern before every BLAS call is threads_blas(T),
// and before every #pragma omp parallel region it's threads_blas(1).
// After the last operation in a function, BLAS is restored to T
// so the caller doesn't need to reset it.
// =====================================================================

static inline void threads_blas(int n) {
    openblas_set_num_threads(n);
    Eigen::setNbThreads(n);
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
        throw std::invalid_argument("NMFModel: k must be in [1, min(n_samples, vocab)].");

    const int T = (cfg_.n_threads > 0) ? cfg_.n_threads : omp_get_max_threads();
    omp_set_num_threads(T);
    threads_blas(T);

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
        const char* solver_name = cfg_.solver == SolverType::SGD ? "SGD (Hogwild!)" : "HALS";
        std::cout << "NMF fit\n"
                  << "  solver="     << solver_name
                  << "  n="          << n
                  << "  vocab="      << vocab
                  << "  k="          << cfg_.k
                  << "  max_iter="   << cfg_.max_iter << "\n"
                  << "  x_mean="     << x_mean
                  << "  init_scale=" << cfg_.scale    << "\n"
                  << "  threads="    << T             << "\n";
        if (cfg_.solver == SolverType::SGD)
            std::cout << "  lr=" << cfg_.lr << "  lr_decay=" << cfg_.lr_decay
                      << "  reg=" << cfg_.reg << "\n";
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
            if (cfg_.shuffle) std::shuffle(perm.begin(), perm.end(), rng);

            threads_blas(1);          // OMP inside; disable BLAS nesting
            update_WH_sgd(X, W, perm, lr);
            threads_blas(T);

            lr *= cfg_.lr_decay;
            ++iters_run_;

            if (iter % cfg_.check_every == 0 || iter == cfg_.max_iter - 1) {
                const float rel_change = relative_h_change(H_old);
                threads_blas(1);
                last_error_ = nnz_rmse(X, W);
                threads_blas(T);

                if (cfg_.verbose)
                    std::cout << "  iter " << iter
                              << "  RMSE_nnz=" << last_error_
                              << "  rel_H_change=" << rel_change
                              << "  lr=" << lr << "\n";

                if (cb && !cb(iter, last_error_)) break;
                if (rel_change < cfg_.tol) {
                    if (cfg_.verbose) std::cout << "  Converged.\n";
                    break;
                }
            }
        }
        fitted_ = true;
        return;
    }

    // ── HALS path ───────────────────────────────────────────────────
    //
    // Per-iteration:
    //   1. HHt = H H^T, XHt = X H^T          (BLAS, T threads)
    //   2. W update                            (blocked HALS)
    //   3. WtW = W^T W, WtX = W^T X           (BLAS, T threads)
    //   4. H update                            (blocked HALS)
    //   5. Convergence check
    //
    // No normalisation between steps 2 and 3.  Normalising W and
    // rescaling H between the two HALS updates destroys the fixed-point:
    // WtW and WtX are computed from the normalised W, but G=WtW*H uses
    // the rescaled H, making the gradient wrong and causing divergence.
    // The reg term prevents scale drift without touching the factors.

    for (int iter = 0; iter < cfg_.max_iter; ++iter) {
        const DenseMat H_old = H_;

        // ── Step 1 ────────────────────────────────────────────────────
        threads_blas(T);

        DenseMat HHt(cfg_.k, cfg_.k);
        HHt.setZero();
        HHt.selfadjointView<Eigen::Lower>().rankUpdate(H_);
        HHt = HHt.selfadjointView<Eigen::Lower>();

        const DenseMat XHt = X * H_.transpose();   // (n, k)

        // ── Step 2: W update ─────────────────────────────────────────
        // Returns with BLAS at T threads.
        update_W_hals<MatType>(W, HHt, XHt, T);

        // ── Step 3 ────────────────────────────────────────────────────
        threads_blas(T);

        DenseMat WtW(cfg_.k, cfg_.k);
        WtW.setZero();
        WtW.selfadjointView<Eigen::Lower>().rankUpdate(W.transpose());
        WtW = WtW.selfadjointView<Eigen::Lower>();

        DenseMat WtX(cfg_.k, vocab);
        if constexpr (std::is_same_v<MatType, SparseMat>)
            WtX = (X.transpose() * W).transpose();
        else
            WtX = W.transpose() * X;

        // ── Step 4: H update ─────────────────────────────────────────
        // Returns with BLAS at T threads.
        update_H_hals<MatType>(WtW, WtX, T);

        ++iters_run_;

        // ── Step 5: Convergence check ─────────────────────────────────
        if (iter % cfg_.check_every == 0 || iter == cfg_.max_iter - 1) {
            const float rel_change = relative_h_change(H_old);

            threads_blas(T);
            last_error_ = reconstruction_error(X, W, WtX, WtW);

            if (cfg_.verbose)
                std::cout << "  iter "         << iter
                          << "  ||X-WH||_F="   << last_error_
                          << "  rel_H_change=" << rel_change << "\n";

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
// Private: update_WH_sgd
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
                const int   j        = it.index();
                const float v        = it.value();
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
// Blocked sequential HALS for W (n x k).
//
// Reduces OMP fork/joins from k (=512) to k/B (=8) by processing B
// components per block, then correcting the gradient matrix G with a
// single BLAS dgemm per block.
//
//   G = W * HHt                       (n x k, one dgemm upfront)
//   for each block b .. b+B:
//     [OMP over n rows — 1 fork/join]
//       for rb in 0..B-1:
//         r = b+rb
//         w_new = max(0, (XHt[:,r] - G[:,r] + W[:,r]*HHt[r,r])
//                        / (HHt[r,r] + reg + eps))
//         W_delta[:,rb] = w_new - W[:,r];  W[:,r] = w_new
//     G += W_delta * HHt[b:b+B,:]     (dgemm — corrects gradient)
//
// Exits with BLAS at T threads.
// =====================================================================

template <typename MatType>
void NMFModel::update_W_hals(DenseMat& W,
                              const DenseMat& HHt,
                              const DenseMat& XHt,
                              int T) const {
    const float eps   = 1e-10f;
    const float reg   = cfg_.reg;
    const int   k     = cfg_.k;
    const int   n     = static_cast<int>(W.rows());
    constexpr int B   = 64;

    threads_blas(T);
    DenseMat G = W * HHt;                 // (n x k)

    for (int b = 0; b < k; b += B) {
        const int bend = std::min(b + B, k);
        const int bsz  = bend - b;

        DenseMat W_delta(n, bsz);

        threads_blas(1);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int rb = 0; rb < bsz; ++rb) {
                const int   r     = b + rb;
                const float denom = HHt(r, r) + reg + eps;
                const float w_old = W(i, r);
                const float w_new = std::max(0.f,
                    (XHt(i, r) - G(i, r) + w_old * HHt(r, r)) / denom);
                W(i, r)        = w_new;
                W_delta(i, rb) = w_new - w_old;
            }
        }

        threads_blas(T);
        G.noalias() += W_delta * HHt.middleRows(b, bsz);
    }
    // Exits with BLAS at T.
}

// =====================================================================
// Private: update_H_hals
//
// Blocked sequential HALS for H (k x vocab), same strategy as W.
//
//   G = WtW * H                        (k x vocab, one dgemm upfront)
//   for each block b .. b+B:
//     [OMP over vocab cols — 1 fork/join]
//       for rb in 0..B-1:
//         r = b+rb
//         h_new = max(0, (WtX[r,:] - G[r,:] + H[r,:]*WtW[r,r])
//                        / (WtW[r,r] + reg + eps))
//         H_delta[rb,:] = h_new - H[r,:];  H[r,:] = h_new
//     G += WtW[:,b:b+B] * H_delta      (dgemm — corrects gradient)
//
// Exits with BLAS at T threads.
// =====================================================================

template <typename MatType>
void NMFModel::update_H_hals(const DenseMat& WtW,
                              const DenseMat& WtX,
                              int T) {
    const float eps   = 1e-10f;
    const float reg   = cfg_.reg;
    const int   k     = cfg_.k;
    const int   vocab = static_cast<int>(H_.cols());
    constexpr int B   = 64;

    Eigen::VectorXf denom(k);
    for (int r = 0; r < k; ++r) denom(r) = WtW(r, r) + reg + eps;

    threads_blas(T);
    DenseMat G = WtW * H_;                // (k x vocab)

    for (int b = 0; b < k; b += B) {
        const int bend = std::min(b + B, k);
        const int bsz  = bend - b;

        DenseMat H_delta(bsz, vocab);

        threads_blas(1);
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < vocab; ++j) {
            for (int rb = 0; rb < bsz; ++rb) {
                const int   r     = b + rb;
                const float h_old = H_(r, j);
                const float h_new = std::max(0.f,
                    (WtX(r, j) - G(r, j) + h_old * WtW(r, r)) / denom(r));
                H_(r, j)        = h_new;
                H_delta(rb, j)  = h_new - h_old;
            }
        }

        threads_blas(T);
        G.noalias() += WtW.middleCols(b, bsz) * H_delta;
    }
    // Exits with BLAS at T.
}

// =====================================================================
// Private: nnz_rmse
// =====================================================================

template <typename MatType>
float NMFModel::nnz_rmse(const MatType& X, const DenseMat& W) const {
    double sum_sq = 0.0;
    long   count  = 0;

    if constexpr (std::is_same_v<MatType, SparseMat>) {
        #pragma omp parallel for reduction(+:sum_sq,count) schedule(dynamic,64)
        for (int i = 0; i < static_cast<int>(X.rows()); ++i)
            for (SparseMat::InnerIterator it(X, i); it; ++it) {
                const float r = it.value() - W.row(i).dot(H_.col(it.index()));
                sum_sq += static_cast<double>(r * r);
                ++count;
            }
    } else {
        #pragma omp parallel for reduction(+:sum_sq,count) schedule(static)
        for (int i = 0; i < static_cast<int>(X.rows()); ++i)
            for (int j = 0; j < static_cast<int>(X.cols()); ++j) {
                const float r = X(i, j) - W.row(i).dot(H_.col(j));
                sum_sq += static_cast<double>(r * r);
                ++count;
            }
    }
    return count > 0 ? static_cast<float>(std::sqrt(sum_sq / count)) : 0.f;
}

// =====================================================================
// Private: relative_h_change
// =====================================================================

float NMFModel::relative_h_change(const DenseMat& H_old) const {
    return (H_ - H_old).norm() / (H_old.norm() + 1e-10f);
}

// =====================================================================
// Private: reconstruction_error
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

    const float cross = WtX.cwiseProduct(H_).sum();

    DenseMat HHt(cfg_.k, cfg_.k);
    HHt.setZero();
    HHt.selfadjointView<Eigen::Lower>().rankUpdate(H_);
    HHt = HHt.selfadjointView<Eigen::Lower>();

    const float WH_sq = (WtW * HHt).trace();
    return std::sqrt(std::max(0.f, X_sq - 2.f * cross + WH_sq));
}

// =====================================================================
// Private: init_factors / init_random
// =====================================================================

void NMFModel::init_factors(int n, int vocab, DenseMat& W, DenseMat& H) const {
    switch (cfg_.init) {
        case InitType::Random: init_random(n, vocab, W, H); return;
        default: throw std::invalid_argument("NMFModel: requested InitType not implemented.");
    }
}

void NMFModel::init_random(int n, int vocab, DenseMat& W, DenseMat& H) const {
    std::mt19937 rng(static_cast<unsigned>(cfg_.seed));
    std::uniform_real_distribution<float> dist(0.f, cfg_.scale);
    W.resize(n, cfg_.k);  for (int i = 0; i < W.size(); ++i) W(i) = dist(rng);
    H.resize(cfg_.k, vocab); for (int i = 0; i < H.size(); ++i) H(i) = dist(rng);
}

// =====================================================================
// Explicit template instantiations
// =====================================================================

template void NMFModel::fit_impl<NMFModel::SparseMat>(const SparseMat&, ProgressCallback);
template void NMFModel::fit_impl<NMFModel::DenseMat>(const DenseMat&, ProgressCallback);

template void NMFModel::update_W_hals<NMFModel::SparseMat>(DenseMat&, const DenseMat&, const DenseMat&, int) const;
template void NMFModel::update_W_hals<NMFModel::DenseMat>(DenseMat&, const DenseMat&, const DenseMat&, int) const;

template void NMFModel::update_H_hals<NMFModel::SparseMat>(const DenseMat&, const DenseMat&, int);
template void NMFModel::update_H_hals<NMFModel::DenseMat>(const DenseMat&, const DenseMat&, int);

template void NMFModel::update_WH_sgd<NMFModel::SparseMat>(const SparseMat&, DenseMat&, const std::vector<int>&, float);
template void NMFModel::update_WH_sgd<NMFModel::DenseMat>(const DenseMat&, DenseMat&, const std::vector<int>&, float);

template float NMFModel::reconstruction_error<NMFModel::SparseMat>(const SparseMat&, const DenseMat&, const DenseMat&, const DenseMat&) const;
template float NMFModel::reconstruction_error<NMFModel::DenseMat>(const DenseMat&, const DenseMat&, const DenseMat&, const DenseMat&) const;

template float NMFModel::nnz_rmse<NMFModel::SparseMat>(const SparseMat&, const DenseMat&) const;
template float NMFModel::nnz_rmse<NMFModel::DenseMat>(const DenseMat&, const DenseMat&) const;