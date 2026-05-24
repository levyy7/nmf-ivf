#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "utils/eigen_utils.h"

class NMFBase {
public:
    enum class Init { Random, Acol };

    struct Config {
        int n_components  = -1;
        int max_iter      = 100;
        double tol        = 1e-4;
        Init init_method = Init::Acol;
        int  acol_p      = 5;
        bool verbose      = false;
        bool compute_error = true;
        unsigned int random_state = 42;
    };

    explicit NMFBase(const Config& cfg);
    virtual ~NMFBase() = default;

    [[nodiscard]] const Mat& components() const { return H_; }

    Mat  fit_transform(const SpMat& X);
    void fit(const SpMat& X);

protected:
    virtual void updateW(const SpMat& X, Mat& W, const Mat& H) const = 0;
    virtual void updateH(const SpMat& X, const Mat& W, Mat& H) const = 0;


    Config cfg;

    Mat W_, H_;
    int    n_components_      = 0;
    int    n_iter_            = 0;
    double reconstruction_err_ = 0.0;

    static constexpr float EPS = 1e-10f;

    void init   (const SpMat& X, int k, Mat& W, Mat& H) const;
    void randomInit(const SpMat& X, int k, Mat& W, Mat& H) const;
    void acolInit  (const SpMat& X, int k, Mat& W, Mat& H) const;

    [[nodiscard]] double computeError(const SpMat& X, const Mat& W, const Mat& H) const;
    [[nodiscard]] static double xMean(const SpMat& X);
};
