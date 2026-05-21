#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>

#include "nmf/base.h"


NMFBase::NMFBase(const Config& cfg)
    : cfg(cfg) {}

Mat NMFBase::fit_transform(const SpMat& X)
{
    const int k = (cfg.n_components <= 0) ? X.cols() : cfg.n_components;
    n_components_ = k;

    randomInit(X, k, W_, H_);

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
                      << "err="    << std::fixed << std::setprecision(2) << reconstruction_err_
                      << "  iter=" << std::setprecision(2) << iter_s  << "s"
                      << "  total=" << std::setprecision(1) << total_s << "s"
                      << "  ("     << std::setprecision(2) << iter_s  << "s/iter)"
                      << "\n";
        }

        if (cfg.tol > 0.0) {
            const double rel = std::abs(prev_err - reconstruction_err_) /
                               std::max(prev_err, 1e-12);
            if (rel <= cfg.tol) {
                if (cfg.verbose)
                    std::cout << "Converged at iter " << iter
                              << " (rel=" << rel << ")\n";
                break;
            }
        }

        prev_err = reconstruction_err_;
    }

    if (cfg.verbose) {
        const double total_s = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t_total).count();
        std::cout << "fit_transform done: " << n_iter_ << " iters"
                  << "  total=" << std::fixed << std::setprecision(1) << total_s << "s"
                  << "  avg="   << std::setprecision(2) << total_s / n_iter_ << "s/iter\n";
    }

    return W_;
}

void NMFBase::fit(const SpMat& X)
{
    fit_transform(X);
}


void NMFBase::randomInit(const SpMat& X, int k, Mat& W, Mat& H) const
{
    const double scale = std::sqrt(std::max(xMean(X), 1e-12) / std::max(k, 1));

    std::mt19937 rng(cfg.random_state);
    std::uniform_real_distribution<float> u(0.0f, 2.0f * static_cast<float>(scale));

    W = Mat::NullaryExpr(X.rows(), k, [&]() { return u(rng); });
    H = Mat::NullaryExpr(k, X.cols(), [&]() { return u(rng); });
}

double NMFBase::computeError(const SpMat& X, const Mat& W, const Mat& H) const
{
    if (!cfg.compute_error) return -1.0;

    const Mat    WtW   = W.transpose() * W;       // k×k
    const Mat    HHt   = H * H.transpose();        // k×k
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