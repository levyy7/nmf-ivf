#include "mu_nmf.h"

MuNMF::MuNMF(const Config& cfg)
    : NMFBase(cfg)
{}

void MuNMF::updateW(const SpMat& X, Mat& W, const Mat& H) const
{
    const Mat HHt = H * H.transpose();
    const Mat XHt = X * H.transpose();
    W.array() *= XHt.array() / (W * HHt).array().max(EPS);
    W = W.array().max(0.f).matrix();
}

void MuNMF::updateH(const SpMat& X, const Mat& W, Mat& H) const
{
    const Mat WtW = W.transpose() * W;
    const Mat WtX = (X.transpose() * W).transpose();
    H.array() *= WtX.array() / (WtW * H).array().max(EPS);
    H = H.array().max(0.f).matrix();
}