#include "nmf_index.h"

// Assume these are the definitions discussed previously
#include "utils/hdf5_loader.h"
#include "utils/hdf5_writer.h"

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <random>
#include <numeric>

#include "backend/adaptive.h"
#include "backend/naive.h"

NMFIndex::NMFIndex(std::unique_ptr<IVFBackend> backend, const Config& cfg)
    : backend_(std::move(backend)),
      cfg_(cfg)
{
    if (!backend_)
    {
        throw std::invalid_argument(
            "NMFIndex requires a valid IVFBackend instance.");
    }
}

std::unique_ptr<NMFIndex> NMFIndex::load(const std::string& path,
                                         BackendFactory backend_factory,
                                         const Config& cfg)
{
    if (cfg.verbose)
    {
        std::cout << "[NMFIndex] Loading index from " << path << "...\n";
    }

    HDF5Loader loader(path);

    // Read the datasets stored within the /index group
    Eigen::MatrixXf H = loader.loadDense<float>("/index/H");
    Eigen::MatrixXi lists = loader.loadDense<int>("/index/lists");

    // Let the caller's factory wrap the concrete implementation
    std::unique_ptr<IVFBackend> backend = backend_factory(H, lists);

    if (cfg.verbose)
    {
        std::cout << "[NMFIndex] Successfully loaded index: "
            << H.rows() << " components, list capacity: " << lists.cols() << "\n";
    }

    return std::make_unique < NMFIndex > (std::move(backend), cfg);
}

void NMFIndex::save_index(const std::string& path) const
{
    if (!is_built())
    {
        throw std::runtime_error("NMFIndex::save — Cannot save an unbuilt index.");
    }

    if (cfg_.verbose)
    {
        std::cout << "[NMFIndex] Saving index to " << path << "...\n";
    }

    HDF5Writer writer(path);
    writer.writeIndex(backend_->components(), backend_->lists());
    writer.writeAttribute("algo", "nmf-ivf");
    writer.writeAttribute("task", "task3");
    writer.writeAttribute("buildtime", build_time_sec_);

    if (cfg_.verbose)
    {
        std::cout << "[NMFIndex] Index saved successfully.\n";
    }
}

void NMFIndex::build(const SpMat& X, const std::unique_ptr<NMFBase>& nmf)
{
    if (!nmf)
    {
        throw std::invalid_argument("NMFIndex::build — NMFBase model is missing.");
    }

    nmf_cfg_ = nmf->cfg;

    auto t0 = std::chrono::steady_clock::now();

    if (cfg_.verbose)
    {
        std::cout << "[NMFIndex] Starting NMF fit and index build...\n";
    }

    SpMat X_fit;
    if (X.rows() > cfg_.sample_size)
    {
        std::vector<int> indices(X.rows());
        std::iota(indices.begin(), indices.end(), 0);

        std::shuffle(indices.begin(), indices.end(),
                     std::mt19937{std::random_device{}()});

        indices.resize(cfg_.sample_size);
        X_fit.resize(cfg_.sample_size, X.cols());

        for (int i = 0; i < cfg_.sample_size; ++i)
        {
            X_fit.row(i) = X.row(indices[i]);
        }

        nmf->fit(X_fit);
    }
    else
    {
        nmf->fit(X);
    }

    backend_->build(X, nmf->components());

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    build_time_sec_ = static_cast<float>(ms / 1000.0);

    if (cfg_.verbose)
    {
        std::cout << "[NMFIndex] Index completely built | total time: " <<
            build_time_sec_ << " sec\n";
    }
}


std::vector<std::vector<IVFBackend::SearchResult>>
NMFIndex::search(const SpMat& queries,
                 const SpMat& X_docs,
                 int top_k,
                 const IVFBackend::SearchParams* params) const
{
    if (!is_built())
    {
        throw std::runtime_error("NMFIndex::search — index is not built");
    }

    return backend_->search(queries, X_docs, top_k, params);
}

std::vector<std::vector<IVFBackend::SearchResult>>
NMFIndex::search_and_save(const std::string& path,
                          const SpMat& queries,
                          const SpMat& X_docs,
                          int top_k,
                          const IVFBackend::SearchParams* params) const
{
    auto t0 = std::chrono::steady_clock::now();

    std::vector<std::vector<IVFBackend::SearchResult>> results = search(
        queries, X_docs, top_k, params);

    auto t1 = std::chrono::steady_clock::now();
    float query_time_sec = std::chrono::duration<float>(t1 - t0).count();

    save_results(path, results, query_time_sec, params);
    return results;
}


bool NMFIndex::is_built() const
{
    return backend_ && backend_->is_built();
}

void NMFIndex::save_results(const std::string& path,
                            const std::vector<std::vector<
                                IVFBackend::SearchResult>>& results,
                            float query_time_sec,
                            const IVFBackend::SearchParams* params) const
{
    if (results.empty()) return;

    if (cfg_.verbose)
    {
        std::cout << "[NMFIndex] Saving search results to " << path << "...\n";
    }

    int n_queries = static_cast<int>(results.size());

    // Determine max_k (in case some queries found fewer results than top_k)
    int max_k = 0;
    for (const auto& res : results)
    {
        max_k = std::max(max_k, static_cast<int>(res.size()));
    }

    // Initialize padding arrays
    Eigen::MatrixXi knns = Eigen::MatrixXi::Constant(n_queries, max_k, -1);
    Eigen::MatrixXf dists = Eigen::MatrixXf::Constant(
        n_queries, max_k, -std::numeric_limits<float>::infinity());

    for (int i = 0; i < n_queries; ++i)
    {
        for (size_t j = 0; j < results[i].size(); ++j)
        {
            knns(i, j) = results[i][j].id;
            dists(i, j) = results[i][j].score;
        }
    }

    std::string build_params = "n_components=" + std::to_string(
            nmf_cfg_.n_components)
        +
        ";max_iter=" + std::to_string(nmf_cfg_.max_iter);

    std::string search_params;
    if (params)
    {
        // Attempt to cast to Adaptive params
        if (auto adapt_p = dynamic_cast<const AdaptiveIVFBackend::SearchParams*>(
            params))
        {
            search_params = "max_misses=" + std::to_string(
                    adapt_p->max_consecutive_misses) +
                ";dr=" + std::to_string(adapt_p->score_drop_ratio);
        }
        // Attempt to cast to Naive params
        else if (auto naive_p = dynamic_cast<const NaiveIVFBackend::SearchParams*>(
            params))
        {
            search_params = "nprobe=" + std::to_string(naive_p->nprobe) +
                ";list_search_factor=" + std::to_string(
                    naive_p->list_search_factor);
        }
    }

    std::string params_string = build_params + ';' + search_params;

    // HDF5Writer opens existing files in RDWR, so it safely appends.
    HDF5Writer writer(path);
    writer.writeKnns(knns, true); // true = auto bumps to 1-based indexing for you
    writer.writeDists(dists);

    writer.writeAttribute("algo", "nmf-ivf");
    writer.writeAttribute("task", "task3");
    writer.writeAttribute("querytime", query_time_sec);
    writer.writeAttribute("buildtime", build_time_sec_);
    writer.writeAttribute("params", params_string);

    if (cfg_.verbose)
    {
        std::cout << "[NMFIndex] Search results successfully saved.\n";
    }
}