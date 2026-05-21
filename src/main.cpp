// main.cpp
#include <iostream>
#include <fstream>
#include <omp.h>
#include <Eigen/Sparse>
#include <cblas.h>
#include <CLI/CLI.hpp>

#include "utils/hdf5_loader.h"
#include "experiments/nmf_evaluation.h"
#include "index/nmf_index.h"
#include "nmf/hals_nmf.h"
#include "nmf/mu_nmf.h"

using namespace std;



struct RunConfig {
    // Dataset
    std::string dataset   = "nq";          // "nq" | "fiqa-dev"
    std::string data_file = "";            // overrides dataset default path
    int         threads   = 8;

    // NMF
    std::string nmf_type   = "hals";
    int    n_components    = 3000;
    double tol             = 1e-4;
    int    max_no_improve  = 10;
    int    max_iter        = 30;
    double forget_factor   = 0.7;
    int    random_state    = 59;

    // Only for HALS
    int w_sweeps   = 1;
    int h_sweeps   = 1;

    // Index
    int  m      = 5000;
    int  nprobe = 20;
    int  sample_size = 300000;

    // Eval
    std::vector<int> recall_at = {10, 30, 100};
    int              eval_k    = 100;
};

// Dataset presets

RunConfig preset(const std::string& name) {
    RunConfig cfg;
    cfg.dataset = name;

    if (name == "nq") {
        cfg.data_file     = "data/nq.h5";
        cfg.n_components  = 3000;
        cfg.max_no_improve= 10;
        cfg.max_iter      = 30;
        cfg.m             = 5000;
        cfg.nprobe        = 20;
    } else if (name == "fiqa-dev") {
        cfg.data_file     = "data/fiqa-dev.h5";
        cfg.n_components  = 512;
        cfg.max_no_improve= 6;
        cfg.max_iter      = 50;
        cfg.m             = 500;
        cfg.nprobe        = 16;
    } else {
        throw std::invalid_argument("Unknown dataset: " + name);
    }
    return cfg;
}


int main(int argc, char** argv) {
    CLI::App app{"NMF Index runner"};

    std::string dataset_name = "nq";
    std::string config_file  = "";
    RunConfig cfg;

    // Top-level flags
    app.add_option("-d,--dataset",   dataset_name,
        "Dataset preset: nq | fiqa-dev")->capture_default_str();
    app.add_option("-t,--threads",   cfg.threads,    "OpenBLAS/OMP threads");

    // NMF overrides
    auto* nmf = app.add_option_group("NMF");
    nmf->add_option("--nmf", cfg.nmf_type, "NMF solver: hals (default)")
        ->capture_default_str();
    nmf->add_option("--n-components",   cfg.n_components);
    nmf->add_option("--tol",            cfg.tol);
    nmf->add_option("--max-iter",       cfg.max_iter);
    nmf->add_option("--forget-factor",  cfg.forget_factor);
    nmf->add_option("--random-state",   cfg.random_state);
    nmf->add_option("--w-sweeps", cfg.w_sweeps, "HALS W sweeps per iter");
    nmf->add_option("--h-sweeps", cfg.h_sweeps, "HALS H sweeps per iter");

    // Index overrides
    auto* idx = app.add_option_group("Index");
    idx->add_option("--m",       cfg.m,      "Number of lists");
    idx->add_option("--nprobe",  cfg.nprobe, "Lists to probe at query time");

    CLI11_PARSE(app, argc, argv);

    // Apply preset first, then overrides that were explicitly set
    RunConfig base = preset(dataset_name);

    if (app.count("--nmf"))          base.nmf_type      = cfg.nmf_type;
    if (app.count("--n-components")) base.n_components  = cfg.n_components;
    if (app.count("--m"))            base.m             = cfg.m;
    if (app.count("--nprobe"))       base.nprobe        = cfg.nprobe;
    if (app.count("--threads"))      base.threads       = cfg.threads;
    if (app.count("--w-sweeps"))     base.w_sweeps      = cfg.w_sweeps;
    if (app.count("--h-sweeps"))     base.h_sweeps      = cfg.h_sweeps;
    if (app.count("--tol"))          base.tol           = cfg.tol;
    if (app.count("--max-iter"))     base.max_iter      = cfg.max_iter;
    if (app.count("--random-state")) base.random_state  = cfg.random_state;
    cfg = base;

    // ── Run ──────────────────────────────────────────────────────────────────
    try {
        openblas_set_num_threads(cfg.threads);
        omp_set_num_threads(cfg.threads);
        std::cout << "Dataset     : " << cfg.dataset       << "\n"
                  << "n_components: " << cfg.n_components  << "\n"
                  << "m / nprobe  : " << cfg.m << " / " << cfg.nprobe << "\n"
                  << "OpenMP max  : " << omp_get_max_threads() << "\n"
                  << Eigen::SimdInstructionSetsInUse()          << "\n";

        #ifdef EIGEN_USE_BLAS
                std::cout << "Eigen: using BLAS backend\n";
        #else
                std::cout << "Eigen: NOT using BLAS backend\n";
        #endif

        HDF5Loader loader(cfg.data_file);

        auto query = loader.loadSparse<float>("otest/queries");
        auto gt    = loader.loadGroundTruth("otest/knns");
        auto train = loader.loadSparse<float>("train");

        std::cout << "train: " << train.rows() << " x " << train.cols()
                  << "  (" << train.nonZeros() << " nnz)\n";

        const NMFIndex::Config idx_cfg{
            cfg.m, cfg.nprobe, cfg.sample_size, /*verbose=*/true
        };

        // ── Build NMF solver ────────────────────────────────────────────────────────
        std::unique_ptr<NMFBase> nmf_solver;

        if (cfg.nmf_type == "hals") {
            HalsNMF::Config hals_cfg;
            hals_cfg.n_components  = cfg.n_components;
            hals_cfg.max_iter      = cfg.max_iter;
            hals_cfg.tol           = cfg.tol;
            hals_cfg.verbose       = true;
            hals_cfg.random_state  = cfg.random_state;
            hals_cfg.w_sweeps      = cfg.w_sweeps;
            hals_cfg.h_sweeps      = cfg.h_sweeps;
            nmf_solver = std::make_unique<HalsNMF>(hals_cfg);
        } else if (cfg.nmf_type == "mu") {
            MuNMF::Config mu_cfg;
            mu_cfg.n_components  = cfg.n_components;
            mu_cfg.max_iter      = cfg.max_iter;
            mu_cfg.tol           = cfg.tol;
            mu_cfg.verbose       = true;
            mu_cfg.random_state  = cfg.random_state;
            nmf_solver = std::make_unique<MuNMF>(mu_cfg);
        } else {
            throw std::invalid_argument("Unknown NMF solver: " + cfg.nmf_type);
        }

        NMFIndex index(std::move(nmf_solver), idx_cfg);
        index.build(train);

        evaluate_nmf_index(index, query, gt,
            cfg.recall_at,
            {1,2,5,10,20,50,index.n_lists()},
            cfg.eval_k);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}