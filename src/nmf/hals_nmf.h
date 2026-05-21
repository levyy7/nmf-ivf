#pragma once

#include "nmf/base.h"

class HalsNMF : public NMFBase {
public:
    struct Config : NMFBase::Config {
        int w_sweeps = 1;
        int h_sweeps = 1;
    };

    explicit HalsNMF(const Config& cfg);

protected:
    int w_sweeps;
    int h_sweeps;

    void updateW(const SpMat& X, Mat& W, const Mat& H) const override;
    void updateH(const SpMat& X, const Mat& W, Mat& H) const override;
};