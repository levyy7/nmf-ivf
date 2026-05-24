// main.cpp
#include <iostream>
#include <fstream>

#include <CLI/CLI.hpp>

#include "run.h"

using namespace std;

// ── Main / CLI Parsing ────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    CLI::App app{"NMF Index Runner"};

    std::string dataset_name = "nq";
    RunConfig   cfg;

    // Top-level flags
    app.add_option("-d,--dataset", dataset_name, "Dataset preset: nq | fiqa-dev")
       ->capture_default_str();
    app.add_option("-t,--threads", cfg.threads, "OpenBLAS/OMP threads");

    // NMF overrides
    auto* nmf_group = app.add_option_group("NMF Model Settings");
    nmf_group->add_option("--nmf",           cfg.nmf_type,    "NMF solver: hals | mu")->capture_default_str();
    nmf_group->add_option("--sample-size",   cfg.sample_size, "Number of training samples to use for NMF (default: 150k)");
    nmf_group->add_option("--n-components",  cfg.n_components);
    nmf_group->add_option("--tol",           cfg.tol);
    nmf_group->add_option("--max-iter",      cfg.max_iter);
    nmf_group->add_option("--forget-factor", cfg.forget_factor);
    nmf_group->add_option("--random-state",  cfg.random_state);
    nmf_group->add_option("--init",          cfg.init_method, "Initialisation: acol (default) | random")->capture_default_str();
    nmf_group->add_option("--acol-p",        cfg.acol_p,      "Rows averaged per component in Acol init (default: 5)");
    nmf_group->add_option("--w-sweeps",      cfg.w_sweeps,    "HALS W sweeps per iter");
    nmf_group->add_option("--h-sweeps",      cfg.h_sweeps,    "HALS H sweeps per iter");
    app.add_flag("--debug",                  cfg.debug,       "Verbose output + error computation each iteration");
    app.add_flag("--evaluate-recall",                  cfg.evaluate_recall,       "Evaluate recall for the built index");

    // Index & Backend overrides
    auto* idx_group = app.add_option_group("Index & Backend Settings");
    idx_group->add_option("--backend", cfg.backend_type, "Backend type: naive")->capture_default_str();
    idx_group->add_option("--m",       cfg.m,            "Number of docs stored per list");
    idx_group->add_option("--nprobe",  cfg.nprobe,       "Number of lists to probe at query time");

    // File Output Options
    auto* io_group = app.add_option_group("File I/O");
    io_group->add_option("--save-index",   cfg.save_index_path,   "Path to save the built index (.h5)");
    io_group->add_option("--save-results", cfg.save_results_path, "Path to save search results (knns & dists .h5)");

    CLI11_PARSE(app, argc, argv);

    // Apply preset base config, then overwrite with explicitly provided CLI arguments
    RunConfig base = preset(dataset_name);

    if (app.count("--nmf"))          base.nmf_type          = cfg.nmf_type;
    if (app.count("--n-components")) base.n_components      = cfg.n_components;
    if (app.count("--backend"))      base.backend_type      = cfg.backend_type;
    if (app.count("--m"))            base.m                 = cfg.m;
    if (app.count("--nprobe"))       base.nprobe            = cfg.nprobe;
    if (app.count("--threads"))      base.threads           = cfg.threads;
    if (app.count("--w-sweeps"))     base.w_sweeps          = cfg.w_sweeps;
    if (app.count("--h-sweeps"))     base.h_sweeps          = cfg.h_sweeps;
    if (app.count("--tol"))          base.tol               = cfg.tol;
    if (app.count("--max-iter"))     base.max_iter          = cfg.max_iter;
    if (app.count("--random-state")) base.random_state      = cfg.random_state;
    if (app.count("--init"))         base.init_method       = cfg.init_method;
    if (app.count("--acol-p"))       base.acol_p            = cfg.acol_p;

    // I/O overrides
    if (app.count("--save-index"))   base.save_index_path   = cfg.save_index_path;
    if (app.count("--save-results")) base.save_results_path = cfg.save_results_path;

    base.debug = app.count("--debug") > 0;

    return run(base);
}