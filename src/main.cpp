#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cblas.h>

#include "nmf_index.h"
#include "clustering/mini_batch_nmf.h"
#include "utils/hdf5_loader.h"
#include "clustering/nmf.h"
#include "experiments/nmf_evaluation.h"

using namespace std;

constexpr int SAMPLE_SIZE = 500000;

template<typename SpMatType>
size_t sparse_memory_bytes(const SpMatType& X) {
    return
        X.nonZeros() * (sizeof(typename SpMatType::Scalar) + sizeof(int)) +
        (X.cols() + 1) * sizeof(int);
}

void print_memory_usage() {
    std::ifstream file("/proc/self/status");
    std::string line;

    while (std::getline(file, line)) {
        if (line.rfind("VmRSS:", 0) == 0 ||  // Resident memory (RAM)
            line.rfind("VmSize:", 0) == 0)   // Virtual memory
        {
            std::cout << line << '\n';
        }
    }
}

int main() {
    try {
        //Eigen::setNbThreads(8);
        openblas_set_num_threads(8);
        std::cout << "OpenMP max threads: " << omp_get_max_threads() << std::endl;
        std::cout << Eigen::SimdInstructionSetsInUse() << std::endl;


        HDF5Loader loader("data/nq.h5");

        auto query = loader.loadSparse<float>("otest/queries");
        std::cout << "query: " << query.rows() << " x " << query.cols()
                  << "  (" << query.nonZeros() << " nnz)\n";

        auto gt = loader.loadGroundTruth("otest/knns");
        std::cout << "gt: " << gt.rows() << " x " << gt.cols() << "\n";

        auto train = loader.loadSparse<float>("train");
        std::cout << "train: " << train.rows() << " x " << train.cols()
                  << "  (" << train.nonZeros() << " nnz)\n";

        size_t bytes = sparse_memory_bytes(train);
        double mb = bytes / (1024.0 * 1024.0);

        std::cout << "Train data memory: " << mb << " MB\n";
        print_memory_usage();

        constexpr MiniBatchNMF::Config cfg(
            /*n_components=*/     512,
            /*batch_size=*/       5000,
            /*tol=*/              1e-4,
            /*max_no_improvement*/6,
            /*max_iter=*/         50,
            /*forget_factor=*/    0.7,
            /*verbose=*/          true,
            /*random_state=*/     55
        );

        constexpr MiniBatchNMF::Config cfg_nq(
            /*n_components=*/     3000,
            /*batch_size=*/       20000,
            /*tol=*/              1e-4,
            /*max_no_improvement*/10,
            /*max_iter=*/         30,
            /*forget_factor=*/    0.7,
            /*verbose=*/          true,
            /*random_state=*/     57
        );

        NMFIndex::Config idx_cfg{
            /*m=*/      500,
            /*nprobe=*/ 16,
                        500000,
            /*verbose=*/ true
        };
        NMFIndex::Config idx_cfg_nq{
            /*m=*/      5000,
            /*nprobe=*/ 20,
                500000,
            /*verbose=*/ true
        };

        NMFIndex index(cfg_nq, idx_cfg_nq);

        index.build(train);


        //NMFModel model(cfg);
        //model.fit(train);

        //NMFIndex index(cfg, NMFIndex::Config{500, 10, true});
        //index.build(train);

        evaluate_nmf_index(
            index,
            query,
            gt,
            {10, 30, 100},
            {1,2,5,10,20,50,index.n_lists()},
            100
        );


    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}