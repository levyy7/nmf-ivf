#include <iostream>
#include <vector>
#include <random>
#include <filesystem>
#include <string>

// Include the header where you placed the RunConfig and run() function
#include "run.h"

namespace fs = std::filesystem;

int main() {
  // The sample sizes to sweep to find the degradation threshold
  std::vector<int> sample_sizes = {5000, 7500, 10000};
  const int runs_per_size = 3;

  std::cout << "Starting Sample Size Degradation Experiment...\n";
  std::cout << "Total runs scheduled: " << sample_sizes.size() * runs_per_size
      << "\n\n";

  int run_counter = 1;

  // Entropy source for dynamic seeds
  std::random_device rd;

  for (int ss : sample_sizes) {
    // 1. Ensure the output directory for this sample size exists
    std::string dir_path = "output/sampling_experiment/sample_" +
                           std::to_string(ss);
    fs::create_directories(dir_path);

    for (int i = 0; i < runs_per_size; ++i) {
      // 2. Generate a dynamic random seed for this specific run
      unsigned int seed = rd();

      // 3. Initialize with preset
      RunConfig cfg = preset("nq");
      cfg.init_method = "acol"; // Fixed init method for consistency

      // 4. Apply fixed parameters from the log
      cfg.nmf_type = "mu";
      cfg.n_components = 5000;
      cfg.max_iter = 20;
      cfg.eval_k = 30;
      cfg.threads = 8;
      cfg.debug = false;
      cfg.evaluate_recall = false; // Evaluates natively in your run() function

      // 5. Apply experimental variables
      cfg.sample_size = ss;
      cfg.random_state = static_cast<int>(seed);

      cfg.backend_type = "adaptive";
      cfg.max_misses = 150;
      cfg.drop_ratio = 0.3f;

      // FIX 2: Match the metric used in your K-sweep
      cfg.recall_at = {30};
      cfg.eval_k = 30;

      // 6. File I/O: Save ONLY the results, ignore the index
      cfg.save_index_path = "";
      cfg.save_results_path = dir_path + "/exp_" + std::to_string(seed) + ".h5";

      // Visual separator for the logs
      std::cout <<
          "\n████████████████████████████████████████████████████████████████\n";
      std::cout << "  RUN " << run_counter++ << "  |  SAMPLE SIZE: " << ss
          << "  |  ITER: " << (i + 1) << "/" << runs_per_size
          << "  |  SEED: " << seed << "\n";
      std::cout << "  OUTPUT: " << cfg.save_results_path << "\n";
      std::cout <<
          "████████████████████████████████████████████████████████████████\n\n";

      // Execute
      run(cfg);
    }
  }

  std::cout << "\n[EXPERIMENT SWEEP COMPLETE]\n";
  return 0;
}