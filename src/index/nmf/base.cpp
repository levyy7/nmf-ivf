#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>

#include "index/nmf/base.h"

NMFBase::NMFBase(const Config& cfg)
    : cfg(cfg) {}

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::string initMethodName(NMFBase::Init m)
{
    switch (m) {
        case NMFBase::Init::Acol:   return "acol";
        case NMFBase::Init::Random: return "random";
        default:                            return "unknown";
    }
}

// ── fit_transform ─────────────────────────────────────────────────────────────

Mat NMFBase::fit_transform(const SpMat& X)
{
    const int k = (cfg.n_components <= 0) ? X.cols() : cfg.n_components;
    n_components_ = k;

    // ── Initialisation ────────────────────────────────────────────────────────
    if (cfg.verbose)
        std::cout << "[init] method=" << initMethodName(cfg.init_method)
                  << "  k=" << k
                  << (cfg.init_method == Init::Acol
                        ? "  acol_p=" + std::to_string(cfg.acol_p)
                        : "")
                  << "\n";

    auto t_init = std::chrono::high_resolution_clock::now();
    init(X, k, W_, H_);

    if (cfg.verbose) {
        const double init_s = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t_init).count();
        std::cout << "[init] done in " << std::fixed << std::setprecision(2)
                  << init_s << "s\n";
    }

    // Reconstruction error right after init — useful to see how much
    // the factorisation actually improves from the starting point.
    if (cfg.verbose && cfg.compute_error) {
        const double init_err = computeError(X, W_, H_);
        std::cout << "[init] reconstruction_err=" << std::fixed
                  << std::setprecision(4) << init_err << "\n";
    }

    // ── Main loop ─────────────────────────────────────────────────────────────
    double prev_err = std::numeric_limits<double>::infinity();
    auto   t_total  = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < cfg.max_iter; ++iter) {
        auto t_iter = std::chrono::high_resolution_clock::now();

        updateW(X, W_, H_);
        updateH(X, W_, H_);

        reconstruction_err_ = computeError(X, W_, H_);
        n_iter_ = iter + 1;

        if (cfg.verbose) {
            auto   now     = std::chrono::high_resolution_clock::now();
            double iter_s  = std::chrono::duration<double>(now - t_iter).count();
            double total_s = std::chrono::duration<double>(now - t_total).count();

            std::cout << "[iter " << std::setw(3) << iter << "] "
                      << "err=" << std::fixed << std::setprecision(4) << reconstruction_err_;

            if (cfg.compute_error && prev_err < std::numeric_limits<double>::infinity()) {
                const double improvement = prev_err - reconstruction_err_;
                const double rel         = improvement / std::max(prev_err, 1e-12);
                std::cout << "  impr=" << std::setprecision(4) << improvement
                          << "  rel="  << std::scientific << std::setprecision(2) << rel
                          << std::defaultfloat;
            }

            std::cout << "  iter="  << std::fixed << std::setprecision(2) << iter_s  << "s"
                      << "  total=" << std::setprecision(1) << total_s << "s"
                      << "\n";
        }

        if (cfg.tol > 0.0) {
            const double rel = std::abs(prev_err - reconstruction_err_) /
                               std::max(prev_err, 1e-12);
            if (rel <= cfg.tol) {
                if (cfg.verbose)
                    std::cout << "[converged] iter=" << iter
                              << "  rel=" << std::scientific << rel
                              << std::defaultfloat << "\n";
                break;
            }
        }

        prev_err = reconstruction_err_;
    }

    if (cfg.verbose) {
        const double total_s = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t_total).count();
        std::cout << "[done] " << n_iter_ << " iters"
                  << "  total=" << std::fixed << std::setprecision(1) << total_s << "s"
                  << "  avg="   << std::setprecision(2) << total_s / n_iter_ << "s/iter"
                  << "  final_err=" << std::setprecision(4) << reconstruction_err_ << "\n";
    }

    return W_;
}

void NMFBase::fit(const SpMat& X)
{
    fit_transform(X);
}

// ── Initialisation ────────────────────────────────────────────────────────────

void NMFBase::init(const SpMat& X, int k, Mat& W, Mat& H) const
{
    switch (cfg.init_method) {
        case Init::Acol:   acolInit(X, k, W, H);   break;
        case Init::Random: randomInit(X, k, W, H); break;
    }
}

void NMFBase::randomInit(const SpMat& X, int k, Mat& W, Mat& H) const
{
    const double scale = std::sqrt(std::max(xMean(X), 1e-12) / std::max(k, 1));

    std::mt19937 rng(cfg.random_state);
    std::uniform_real_distribution<float> u(0.0f, 2.0f * static_cast<float>(scale));

    W = Mat::NullaryExpr(X.rows(), k, [&]() { return u(rng); });
    H = Mat::NullaryExpr(k, X.cols(), [&]() { return u(rng); });
}

void NMFBase::acolInit(const SpMat& X, int k, Mat& W, Mat& H) const
{
    const int n = X.rows();
    const int p = std::max(1, cfg.acol_p);

    std::mt19937 rng(cfg.random_state);
    std::uniform_int_distribution<int> row_dist(0, n - 1);

    H = Mat::Zero(k, X.cols());
    for (int r = 0; r < k; ++r) {
        for (int s = 0; s < p; ++s) {
            const int idx = row_dist(rng);
            for (SpMat::InnerIterator it(X, idx); it; ++it)
                H(r, it.col()) += it.value();
        }
        H.row(r) /= static_cast<float>(p);
    }

    const float scale = static_cast<float>(
        std::sqrt(std::max(xMean(X), 1e-12) / std::max(k, 1)));

    std::uniform_real_distribution<float> u(0.0f, scale);
    W = Mat::NullaryExpr(X.rows(), k, [&]() { return u(rng); });
}

// ── Error ─────────────────────────────────────────────────────────────────────

double NMFBase::computeError(const SpMat& X, const Mat& W, const Mat& H) const
{
    if (!cfg.compute_error) return -1.0;

    const Mat WtW   = W.transpose() * W;
    const Mat HHt   = H * H.transpose();
    const double wh_sq = WtW.cwiseProduct(HHt).sum();

    double xsq = 0.0, cross = 0.0;

    #pragma omp parallel for reduction(+:xsq, cross) schedule(dynamic, 64)
    for (int i = 0; i < X.outerSize(); ++i) {
        const auto w_i = W.row(i);
        for (SpMat::InnerIterator it(X, i); it; ++it) {
            const float v = it.value();
            xsq   += v * v;
            cross += v * w_i.dot(H.col(it.col()));
        }
    }

    return std::sqrt(std::max(0.0, xsq - 2.0 * cross + wh_sq));
}

double NMFBase::xMean(const SpMat& X)
{
    double s = 0.0;
    for (int r = 0; r < X.outerSize(); ++r)
        for (SpMat::InnerIterator it(X, r); it; ++it)
            s += it.value();

    return s / std::max<std::ptrdiff_t>(
        1, static_cast<std::ptrdiff_t>(X.rows()) * X.cols());
}