#include "hals_nmf.h"

#include <cblas.h>


HalsNMF::HalsNMF(const Config& cfg)
    : NMFBase(cfg),
    w_sweeps(cfg.w_sweeps),
    h_sweeps(cfg.h_sweeps)
{}


void HalsNMF::updateW(const SpMat& X, Mat& W, const Mat& H) const
{
    const int n = X.rows();
    const int k = H.rows();
    const Mat G = H * H.transpose();   // (k×k) ColMajor ✓

    openblas_set_num_threads(1);
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n; ++i) {

        // c_i = X_i · Hᵀ — iterate only over sparse nonzeros
        Eigen::VectorXf c = Eigen::VectorXf::Zero(k);
        for (SpMat::InnerIterator it(X, i); it; ++it)
            c.noalias() += it.value() * H.col(it.col());

        Eigen::VectorXf w  = W.row(i).transpose();
        Eigen::VectorXf Gw = G * w;    // residual tracker, updated incrementally

        for (int s = 0; s < w_sweeps; ++s) {
            for (int r = 0; r < k; ++r) {
                const float numer  = c(r) - Gw(r) + G(r, r) * w(r);
                const float new_w  = std::max(0.f, numer / (G(r, r) + EPS));
                const float delta  = new_w - w(r);

                if (delta != 0.f) {
                    Gw.noalias() += delta * G.col(r);   // O(k), contiguous ✓
                    w(r) = new_w;
                }
            }
        }

        W.row(i) = w.transpose();
    }
    openblas_set_num_threads(8);
}

void HalsNMF::updateH(const SpMat& X, const Mat& W, Mat& H) const
{
    const Mat A = W.transpose() * W;                     // (k×k) ColMajor ✓
    const Mat B = (X.transpose() * W).transpose();       // (k×f)

    const int k = H.rows();
    const int f = H.cols();

    openblas_set_num_threads(1);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < f; ++j) {
        Eigen::VectorXf h  = H.col(j);
        Eigen::VectorXf Ah = A * h;    // residual tracker, updated incrementally
        const auto      b  = B.col(j);

        for (int s = 0; s < h_sweeps; ++s) {
            for (int r = 0; r < k; ++r) {
                const float numer  = b(r) - Ah(r) + A(r, r) * h(r);
                const float new_h  = std::max(0.f, numer / (A(r, r) + EPS));
                const float delta  = new_h - h(r);

                if (delta != 0.f) {
                    Ah.noalias() += delta * A.col(r);   // O(k), contiguous ✓
                    h(r) = new_h;
                }
            }
        }

        H.col(j) = h;
    }
    openblas_set_num_threads(8);
}
