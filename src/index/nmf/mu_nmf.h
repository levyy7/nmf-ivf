#pragma once

#include "index/nmf/base.h"


class MuNMF : public NMFBase {
public:
    struct Config : NMFBase::Config {
        // No extra hyperparameters
    };

    explicit MuNMF(const Config& cfg);

protected:
    void updateW(const SpMat& X, Mat& W, const Mat& H) const override;
    void updateH(const SpMat& X, const Mat& W, Mat& H) const override;
};