#include "hals_nmf.h"

HalsNMF::HalsNMF(const Config& cfg)
  : NMFBase(cfg) {
}

void HalsNMF::updateW(const SpMat& X, Mat& W, const Mat& H) const {
  // Precompute constant matrices for this update step
  const Mat HHt = H * H.transpose(); // Size: K x K
  const Mat XHt = X * H.transpose(); // Size: N x K (Sparse * Dense)

  // HALS updates MUST be strictly sequential.
  // Do not use OpenMP here; W.col(k) relies on the updated W.col(j) for j < k.
  for (int k = 0; k < W.cols(); ++k) {
    // 1. Calculate the unconstrained least-squares step for column k.
    // Mathematically: XHt_{:, k} - sum_{j != k} W_{:, j} * HHt_{j, k}
    // We optimize this by subtracting the full sum (W * HHt) and adding back the k-th term.
    Eigen::VectorXf wk = XHt.col(k) - (W * HHt.col(k)) + (W.col(k) * HHt(k, k));

    // 2. Divide by the diagonal scaling factor and project negative values to 0
    W.col(k) = (wk.array() / std::max(HHt(k, k), EPS)).max(0.0f).matrix();
  }
}

void HalsNMF::updateH(const SpMat& X, const Mat& W, Mat& H) const {
  // Precompute constant matrices for this update step
  const Mat WtW = W.transpose() * W; // Size: K x K

  // (X^T * W)^T is the most efficient way to multiply Dense^T * Sparse in Eigen
  const Mat WtX = (X.transpose() * W).transpose(); // Size: K x M

  // Updates MUST be strictly sequential.
  for (int k = 0; k < H.rows(); ++k) {
    // 1. Calculate the unconstrained least-squares step for row k.
    // Mathematically: WtX_{k, :} - sum_{j != k} WtW_{k, j} * H_{j, :}
    Eigen::RowVectorXf hk = WtX.row(k) - (WtW.row(k) * H) + (
                              H.row(k) * WtW(k, k));

    // 2. Divide by diagonal scaling factor and project to >= 0
    H.row(k) = (hk.array() / std::max(WtW(k, k), EPS)).max(0.0f).matrix();
  }
}