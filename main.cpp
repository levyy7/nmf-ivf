#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "run.h"

using namespace std;
using json = nlohmann::json;

std::string getCurrentDate() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
  return ss.str();
}

int main(int argc, char** argv) {
  CLI::App app{"NMF Index Runner"};

  RunConfig cli_cfg;

  app.add_option("--input", cli_cfg.input_path,
                 "Path to the dataset .h5 file")->required();
  app.add_option("--task-description", cli_cfg.task_desc_path,
                 "Path to the config.json file")->required();
  app.add_option("--output", cli_cfg.output_dir,
                 "Directory to save output files")->required();
  app.add_option("-t,--threads", cli_cfg.threads, "OpenBLAS/OMP threads");
  auto* nmf_group = app.add_option_group("NMF Model Settings");
  nmf_group->add_option("--nmf", cli_cfg.nmf_type, "NMF solver: hals | mu");
  nmf_group->add_option("--sample-size", cli_cfg.sample_size,
                        "NMF train samples");
  nmf_group->add_option("--n-components", cli_cfg.n_components);
  nmf_group->add_option("--tol", cli_cfg.tol);
  nmf_group->add_option("--max-iter", cli_cfg.max_iter);
  nmf_group->add_option("--random-state", cli_cfg.random_state);
  nmf_group->add_option("--init", cli_cfg.init_method, "Init: acol | random");
  nmf_group->add_option("--acol-p", cli_cfg.acol_p);
  app.add_flag("--debug", cli_cfg.debug, "Verbose output + error computation");

  // Index & Backend overrides
  auto* idx_group = app.add_option_group("Index & Backend Settings");
  idx_group->add_option("--backend", cli_cfg.backend_type);
  idx_group->add_option("--m", cli_cfg.m);
  idx_group->add_option("--nprobe", cli_cfg.nprobe);

  idx_group->add_option("--max-misses", cli_cfg.max_misses,
                        "Grid search over misses (e.g., --max-misses 60 80 100)")
           ->expected(-1);
  idx_group->add_option("--drop-ratio", cli_cfg.drop_ratio,
                        "Grid search over drop ratios (e.g., --drop-ratio 0.1 0.2)")
           ->expected(-1);

  // File Output Options
  auto* io_group = app.add_option_group("File I/O");
  io_group->add_flag("--no-save-index", cli_cfg.skip_save_index);
  io_group->add_flag("--no-save-results", cli_cfg.skip_save_results);

  CLI11_PARSE(app, argc, argv);

  // 1. Parse JSON Config
  std::ifstream f(cli_cfg.task_desc_path);
  if (!f.is_open()) {
    std::cerr << "Fatal Error: Cannot open JSON: " << cli_cfg.task_desc_path <<
        "\n";
    return 1;
  }

  json task_config;
  f >> task_config;

  std::string ds_name = task_config.value("dataset_name", "nq");
  RunConfig final_cfg = preset(ds_name);

  final_cfg.input_path = cli_cfg.input_path;
  final_cfg.task_desc_path = cli_cfg.task_desc_path;
  final_cfg.output_dir = cli_cfg.output_dir;
  final_cfg.task_id = task_config.value("task", "unknown-task");

  final_cfg.h5_train_path = task_config.value("data", "train");
  final_cfg.h5_queries_path = task_config.value("queries", "otest/queries");
  final_cfg.eval_k = task_config.value("k", 30);

  std::filesystem::create_directories(final_cfg.output_dir);

  std::string date_str = getCurrentDate();
  std::string filename = final_cfg.task_id + "-" + final_cfg.dataset_name + "-"
                         + date_str + ".h5";
  final_cfg.output_path = final_cfg.output_dir + "/" + filename;

  // Map I/O Skips
  final_cfg.skip_save_index = cli_cfg.skip_save_index;
  final_cfg.skip_save_results = cli_cfg.skip_save_results;

  // Merge CLI Overrides
  if (app.count("--nmf")) final_cfg.nmf_type = cli_cfg.nmf_type;
  if (app.count("--sample-size")) final_cfg.sample_size = cli_cfg.sample_size;
  if (app.count("--n-components"))
    final_cfg.n_components = cli_cfg.n_components;
  if (app.count("--backend")) final_cfg.backend_type = cli_cfg.backend_type;
  if (app.count("--m")) final_cfg.m = cli_cfg.m;
  if (app.count("--nprobe")) final_cfg.nprobe = cli_cfg.nprobe;
  if (app.count("--max-misses")) final_cfg.max_misses = cli_cfg.max_misses;
  if (app.count("--drop-ratio")) final_cfg.drop_ratio = cli_cfg.drop_ratio;
  if (app.count("--threads")) final_cfg.threads = cli_cfg.threads;
  if (app.count("--tol")) final_cfg.tol = cli_cfg.tol;
  if (app.count("--max-iter")) final_cfg.max_iter = cli_cfg.max_iter;
  if (app.count("--random-state"))
    final_cfg.random_state = cli_cfg.random_state;
  if (app.count("--init")) final_cfg.init_method = cli_cfg.init_method;
  if (app.count("--acol-p")) final_cfg.acol_p = cli_cfg.acol_p;
  if (app.count("--debug")) final_cfg.debug = cli_cfg.debug;

  return run(final_cfg);
}