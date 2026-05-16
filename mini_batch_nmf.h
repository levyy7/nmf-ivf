#pragma once

// =============================================================================
// MiniBatchNMF — C++ / Eigen implementation
//
// Mirrors the sklearn MiniBatchNMF API with:
//   • Random initialisation only
//   • Frobenius (L2) loss only
//   • No regularisation (alpha_W / alpha_H / l1_ratio)
//   • Sparse input matrix via Eigen::SparseMatrix<double, RowMajor>
//
// Dependencies: Eigen 3.x  (header-only)
//   CMake example:
//     find_package(Eigen3 REQUIRED)
//     target_link_libraries(my_target PRIVATE Eigen3::Eigen)
//
// Public API:
//   MiniBatchNMF nmf(n_components, batch_size, tol, max_no_improvement,
//                    max_iter, forget_factor, verbose, random_state);
//   Mat W = nmf.fit_transform(X);   // X: SpMat (n_samples × n_features)
//   Mat W = nmf.transform(X_new);
//   nmf.partial_fit(X_chunk);       // streaming / out-of-core
//   Mat H = nmf.components_;        // (n_components × n_features)
// =============================================================================

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

// RowMajor lets us slice contiguous row ranges cheaply.
using SpMat = Eigen::SparseMatrix<float, Eigen::RowMajor>;
using Mat   = Eigen::MatrixXf;

// =============================================================================
class MiniBatchNMF {
public:
    // ---- Configuration struct -----------------------------------------------
    struct Config {
        int          n_components       = -1;
        int          batch_size         = 1024;
        double       tol                = 1e-4;
        int          max_no_improvement = 10;
        int          max_iter           = 200;
        double       forget_factor      = 0.7;
        bool         verbose            = false;
        unsigned int random_state       = 42;
    };

    // ---- Hyperparameters (public, set via constructor) ----------------------
    int          n_components;       // -1  → auto (= n_features)
    int          batch_size;
    double       tol;
    int          max_no_improvement;
    int          max_iter;
    double       forget_factor;
    bool         verbose;
    unsigned int random_state;

    // ---- Fitted attributes --------------------------------------------------
    Mat    components_;              // H  (k × n_features)
    int    n_components_;
    double reconstruction_err_;
    int    n_iter_;
    int    n_steps_;

    // -------------------------------------------------------------------------
    explicit MiniBatchNMF(
        int          n_components        = -1,
        int          batch_size          = 1024,
        double       tol                 = 1e-4,
        int          max_no_improvement  = 10,
        int          max_iter            = 200,
        double       forget_factor       = 0.7,
        bool         verbose             = false,
        unsigned int random_state        = 42
    )
    : n_components(n_components), batch_size(batch_size), tol(tol),
      max_no_improvement(max_no_improvement), max_iter(max_iter),
      forget_factor(forget_factor), verbose(verbose),
      random_state(random_state),
      n_components_(0), reconstruction_err_(0), n_iter_(0), n_steps_(0),
      _k(0), _batch_sz(0), _rho(0),
      _ewa_cost(0), _ewa_init(false),
      _ewa_min(0),  _ewa_min_init(false),
      _no_improvement(0), _fitted(false)
    {}

    // Construct from Config struct
    explicit MiniBatchNMF(const Config& cfg)
    : MiniBatchNMF(cfg.n_components, cfg.batch_size, cfg.tol,
                   cfg.max_no_improvement, cfg.max_iter, cfg.forget_factor,
                   cfg.verbose, cfg.random_state)
    {}

    // =========================================================================
    // fit_transform(X)
    //   Factorises X ≈ W·H, stores H in components_, returns W.
    //   X must be non-negative.
    // =========================================================================
    Mat fit_transform(const SpMat& X) {
        _setupParams(X);

        Mat W, H;
        _randomInit(X, W, H);

        // Online accumulators for H updates
        //   _A  (k × f):  Σ  ρ^{t} · Wᵀ Xbatch       (numerator)
        //   _B  (k × k):  Σ  ρ^{t} · Wᵀ W             (denominator factor)
        _A = Mat::Zero(_k, X.cols());
        _B = Mat::Zero(_k, _k);

        _resetConvergenceState();

        Mat H_old = H;
        int n = X.rows();
        int n_steps_per_iter = (n + _batch_sz - 1) / _batch_sz;
        int n_steps_total    = max_iter * n_steps_per_iter;
        int step = 0;
        bool converged = false;

        for (int iter = 0; iter < max_iter && !converged; ++iter) {
            for (int bi = 0; bi < n_steps_per_iter && !converged; ++bi, ++step) {
                int start = bi * _batch_sz;
                int end   = std::min(start + _batch_sz, n);

                SpMat Xb = _sliceRows(X, start, end);
                // W rows for this batch (copy in, copy out)
                Mat Wb = W.middleRows(start, end - start);

                double cost = _minibatchStep(Xb, Wb, H, /*update_H=*/true);
                W.middleRows(start, end - start) = Wb;

                if (_checkConvergence(Xb, cost, H, H_old, n, step, n_steps_total))
                    converged = true;

                H_old = H;
            }
        }

        n_steps_      = step;
        n_iter_       = (step + n_steps_per_iter - 1) / n_steps_per_iter;
        n_components_ = _k;
        components_   = H;
        _fitted       = true;

        reconstruction_err_ = _fullReconstructionError(X, W, H, n_steps_per_iter, n);

        if (!converged && tol > 0.0)
            std::cerr << "Warning: reached max_iter=" << max_iter
                      << ". Consider increasing it.\n";
        return W;
    }

    // =========================================================================
    // transform(X)
    //   Given fitted H = components_, solve for W minimising ||X - WH||_F.
    // =========================================================================
    Mat transform(const SpMat& X) const {
        if (!_fitted) throw std::runtime_error("MiniBatchNMF: call fit_transform first.");
        return _solveW(X, components_, max_iter);
    }

    // =========================================================================
    // partial_fit(X)
    //   Online / out-of-core update.  Call repeatedly on successive chunks.
    //   On the first call the model is initialised from X.
    //   Returns *this for chaining.
    // =========================================================================
    MiniBatchNMF& partial_fit(const SpMat& X) {
        int n = X.rows(), f = X.cols();

        if (!_fitted) {
            // First call — initialise
            _k        = (n_components <= 0) ? f : n_components;
            _batch_sz = std::min(batch_size, n);
            _rho      = std::pow(forget_factor, (double)_batch_sz / n);

            Mat W_dummy;
            _randomInit(X, W_dummy, components_);
            _A = Mat::Zero(_k, f);
            _B = Mat::Zero(_k, _k);
            n_components_ = _k;
            n_steps_      = 0;
            _fitted       = true;
        }

        Mat H  = components_;
        // Solve W for this chunk with fixed H
        Mat Wb = _solveW(X, H, /*max_it=*/30);

        // Online H update  (same accumulator logic as in _minibatchStep)
        _A = _rho * _A + (X.transpose() * Wb).transpose();  // (k × f)
        _B = _rho * _B + Wb.transpose() * Wb;               // (k × k)
        Mat BH = _B * H;                                     // (k × f)
        H = (H.array() * _A.array() / BH.array().max(EPS)).matrix();
        H = H.array().max(0.0).matrix();

        components_ = H;
        ++n_steps_;
        return *this;
    }

private:
    // ---- Internal state -----------------------------------------------------
    static constexpr double EPS = 1e-10;

    int    _k;           // actual n_components used
    int    _batch_sz;
    double _rho;         // effective forget factor per batch

    Mat _A;              // numerator accumulator   (k × f)
    Mat _B;              // denominator accumulator (k × k)

    double _ewa_cost;    // exponentially weighted average cost
    bool   _ewa_init;
    double _ewa_min;
    bool   _ewa_min_init;
    int    _no_improvement;
    bool   _fitted;

    // ---- Setup --------------------------------------------------------------

    void _setupParams(const SpMat& X) {
        int n = X.rows();
        _k        = (n_components <= 0) ? X.cols() : n_components;
        _batch_sz = std::min(batch_size, n);
        _rho      = std::pow(forget_factor, (double)_batch_sz / n);
    }

    void _resetConvergenceState() {
        _ewa_init = _ewa_min_init = false;
        _no_improvement = 0;
        _ewa_cost = _ewa_min = 0.0;
    }

    // ---- Data helpers -------------------------------------------------------

    // Extract rows [start, end) from a RowMajor sparse matrix — O(nnz in range).
    static SpMat _sliceRows(const SpMat& X, int start, int end) {
        SpMat out(end - start, X.cols());
        std::vector<Eigen::Triplet<double>> trips;
        for (int r = start; r < end; ++r)
            for (SpMat::InnerIterator it(X, r); it; ++it)
                trips.emplace_back(r - start, it.col(), it.value());
        out.setFromTriplets(trips.begin(), trips.end());
        return out;
    }

    // Sum of squared non-zero values (= ||X||_F² for non-negative X).
    static double _sparseNormSq(const SpMat& X) {
        double s = 0.0;
        for (int r = 0; r < X.outerSize(); ++r)
            for (SpMat::InnerIterator it(X, r); it; ++it)
                s += it.value() * it.value();
        return s;
    }

    // Mean of all elements (including implicit zeros).
    static double _xMean(const SpMat& X) {
        double s = 0.0;
        for (int r = 0; r < X.outerSize(); ++r)
            for (SpMat::InnerIterator it(X, r); it; ++it)
                s += it.value();
        return s / (double)(X.rows() * X.cols());
    }

    // 0.5 · ||Xb − Wb H||_F² / batch_size, computed without forming X_dense.
    // Uses:  ||Xb − WH||² = ||Xb||² − 2⟨Xb, WH⟩ + ||WH||²
    static double _frobeniusCost(const SpMat& Xb, const Mat& Wb, const Mat& H) {
        double xsq  = _sparseNormSq(Xb);
        Mat    WH   = Wb * H;               // (batch × f), dense
        double cross = 0.0;
        for (int r = 0; r < Xb.outerSize(); ++r)
            for (SpMat::InnerIterator it(Xb, r); it; ++it)
                cross += it.value() * WH(it.row(), it.col());
        return 0.5 * (xsq - 2.0 * cross + WH.squaredNorm()) / Wb.rows();
    }

    // ---- Initialisation -----------------------------------------------------

    // Random init: W, H ~ Uniform[0, 2·scale], scale = sqrt(mean(X) / k)
    void _randomInit(const SpMat& X, Mat& W, Mat& H) const {
        double scale = std::sqrt(_xMean(X) / std::max(_k, 1));
        std::mt19937 rng(random_state);
        std::uniform_real_distribution<double> u(0.0, 2.0 * scale);

        W = Mat::NullaryExpr(X.rows(), _k,  [&](){ return u(rng); });
        H = Mat::NullaryExpr(_k, X.cols(),  [&](){ return u(rng); });
    }

    // ---- W update / solve ---------------------------------------------------

    // One multiplicative update of W (Frobenius):
    //   W ← W ⊙ (Xb Hᵀ) / (W H Hᵀ + ε)
    //
    // Xb is sparse (batch × f); H is dense (k × f).
    // Eigen supports  SpMat * Dense  directly when SpMat is on the left.
    static void _mulUpdateW(const SpMat& Xb, Mat& W, const Mat& H) {
        Mat HHt  = H * H.transpose();    // (k × k)
        Mat XbHt = Xb * H.transpose();   // (batch × k)  sparse·dense ✓
        Mat denom = W * HHt;             // (batch × k)
        W = (W.array() * XbHt.array() / denom.array().max(EPS)).matrix();
        W = W.array().max(0.0).matrix();
    }

    // Iteratively solve for W with H fixed (used by transform / partial_fit).
    // Pre-computes X·Hᵀ and H·Hᵀ once, then loops.
    Mat _solveW(const SpMat& X, const Mat& H, int max_it) const {
        double avg = std::sqrt(_xMean(X) / std::max(_k, 1));
        Mat W     = Mat::Constant(X.rows(), _k, avg);
        Mat W_old = W;

        Mat HHt = H * H.transpose();      // (k × k)
        Mat XHt = X * H.transpose();      // (n × k)  sparse·dense ✓

        for (int it = 0; it < max_it; ++it) {
            Mat denom = W * HHt;
            W = (W.array() * XHt.array() / denom.array().max(EPS)).matrix();
            W = W.array().max(0.0).matrix();

            if (tol > 0.0) {
                double wn = W.norm();
                if (wn > EPS && (W - W_old).norm() / wn <= tol) break;
            }
            W_old = W;
        }
        return W;
    }

    // ---- H update (online) --------------------------------------------------

    // Perform one full mini-batch step:
    //   1. Update Wb  (one multiplicative W step)
    //   2. Accumulate A, B
    //   3. Update H   (one multiplicative H step using accumulated A, B)
    // Returns mean batch Frobenius cost (before H update).
    double _minibatchStep(const SpMat& Xb, Mat& Wb, Mat& H, bool update_H) {
        _mulUpdateW(Xb, Wb, H);
        double cost = _frobeniusCost(Xb, Wb, H);

        if (update_H) {
            // Accumulate (with forget factor):
            //   A = ρ·A + Wᵀ·Xb    →  use (Xbᵀ·W)ᵀ so sparse is always on left
            //   B = ρ·B + Wᵀ·W
            _A = _rho * _A + (Xb.transpose() * Wb).transpose();  // (k × f)
            _B = _rho * _B + Wb.transpose() * Wb;                 // (k × k)

            // H update:  H ← H ⊙ A / (B·H + ε)
            Mat BH = _B * H;                                       // (k × f)
            H = (H.array() * _A.array() / BH.array().max(EPS)).matrix();
            H = H.array().max(0.0).matrix();
        }
        return cost;
    }

    // ---- Convergence --------------------------------------------------------

    bool _checkConvergence(
        const SpMat& Xb, double cost,
        const Mat& H, const Mat& H_old,
        int n_samples, int step, int n_steps
    ) {
        int s    = step + 1;
        int bsz  = Xb.rows();

        // Step 1 is skipped (H not yet updated)
        if (s == 1) {
            if (verbose) _logStep(s, n_steps, cost, -1.0);
            return false;
        }

        // Exponentially weighted average of cost
        if (!_ewa_init) { _ewa_cost = cost; _ewa_init = true; }
        else {
            double alpha = std::min((double)bsz / (n_samples + 1.0), 1.0);
            _ewa_cost = (1.0 - alpha) * _ewa_cost + alpha * cost;
        }

        if (verbose) _logStep(s, n_steps, cost, _ewa_cost);

        // ---- Criterion 1: relative H change ---------------------------------
        double hn = H.norm();
        if (tol > 0.0 && hn > EPS && (H - H_old).norm() / hn <= tol) {
            if (verbose) std::cout << "  Converged (H change) at step "
                                   << s << "/" << n_steps << "\n";
            return true;
        }

        // ---- Criterion 2: EWA cost plateau ----------------------------------
        if (!_ewa_min_init || _ewa_cost < _ewa_min) {
            _ewa_min = _ewa_cost; _ewa_min_init = true; _no_improvement = 0;
        } else {
            ++_no_improvement;
        }
        if (max_no_improvement > 0 && _no_improvement >= max_no_improvement) {
            if (verbose) std::cout << "  Converged (no improvement) at step "
                                   << s << "/" << n_steps << "\n";
            return true;
        }
        return false;
    }

    static void _logStep(int s, int n_steps, double cost, double ewa) {
        std::cout << "[step " << s << "/" << n_steps << "] cost=" << cost;
        if (ewa >= 0.0) std::cout << "  ewa=" << ewa;
        std::cout << "\n";
    }

    // ---- Reconstruction error -----------------------------------------------

    // ||X − WH||_F over the full dataset, computed batch-by-batch to avoid
    // materialising the dense (n × f) product.
    double _fullReconstructionError(
        const SpMat& X, const Mat& W, const Mat& H,
        int n_steps_per_iter, int n
    ) const {
        double err_sq = 0.0;
        for (int bi = 0; bi < n_steps_per_iter; ++bi) {
            int start = bi * _batch_sz;
            int end   = std::min(start + _batch_sz, n);
            SpMat Xb  = _sliceRows(X, start, end);
            Mat   Wb  = W.middleRows(start, end - start);
            Mat   WH  = Wb * H;
            double xsq = _sparseNormSq(Xb), cross = 0.0;
            for (int r = 0; r < Xb.outerSize(); ++r)
                for (SpMat::InnerIterator it(Xb, r); it; ++it)
                    cross += it.value() * WH(it.row(), it.col());
            err_sq += xsq - 2.0 * cross + WH.squaredNorm();
        }
        return std::sqrt(std::max(0.0, err_sq));
    }
};