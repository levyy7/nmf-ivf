#pragma once

#include <string>
#include <vector>
#include <variant>
#include <stdexcept>
#include <functional>

#include <hdf5.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

// Loads sparse or dense matrices from an HDF5 file.
//
// Sparse layout (scipy.sparse / h5py convention — stored as a group):
//   <group>/data            – non-zero values
//   <group>/indices         – column indices (int32 or int64)
//   <group>/indptr          – row pointers   (int32 or int64)
//   <group>.attrs["shape"]  – [nrows, ncols]
//
// Dense layout (plain 2D dataset):
//   <dataset>               – row-major 2-D array of shape [nrows, ncols]

class HDF5Loader {
public:
    template <typename Scalar>
    using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;

    template <typename Scalar>
    using DenseMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    template <typename Scalar>
    using MatrixVariant = std::variant<DenseMatrix<Scalar>, SparseMatrix<Scalar>>;

    // ── RAII handle for any HDF5 resource ───────────────────────────
    struct H5Handle {
        hid_t id = -1;
        std::function<herr_t(hid_t)> closer;

        H5Handle(hid_t id, std::function<herr_t(hid_t)> closer)
            : id(id), closer(std::move(closer)) {}

        ~H5Handle() { if (id >= 0) closer(id); }

        // Non-copyable, movable
        H5Handle(const H5Handle&)            = delete;
        H5Handle& operator=(const H5Handle&) = delete;

        operator hid_t() const { return id; }
    };

    // ────────────────────────────────────────────────────────────────
    explicit HDF5Loader(const std::string& filepath) {
        file_ = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_ < 0)
            throw std::runtime_error("Failed to open HDF5 file: " + filepath);
    }

    ~HDF5Loader() { if (file_ >= 0) H5Fclose(file_); }

    HDF5Loader(const HDF5Loader&)            = delete;
    HDF5Loader& operator=(const HDF5Loader&) = delete;

    // ── Sparse load — direct CSR construction, zero extra copies ────
    template <typename Scalar>
    SparseMatrix<Scalar> loadSparse(const std::string& path) const {
        H5Handle grp{ H5Gopen2(file_, path.c_str(), H5P_DEFAULT), H5Gclose };
        if (grp.id < 0)
            throw std::runtime_error("Failed to open group: " + path);

        auto [nrows, ncols] = readShapeAttr(grp.id, path);

        // ── Read nnz and indptr first (cheap) ───────────────────────────
        auto csr_rowptr = read1D<int64_t>(grp.id, "indptr");
        const int64_t nnz = csr_rowptr.back();

        // ── Allocate the final Eigen matrix immediately ──────────────────
        // Then read values and indices DIRECTLY into Eigen's own buffers.
        // This halves peak memory: no intermediate vectors sitting alongside
        // the final matrix.
        SparseMatrix<Scalar> mat(nrows, ncols);
        mat.reserve(nnz);

        // Eigen compressed storage pointers — write directly into them
        // by reading HDF5 straight into the matrix's internal arrays.
        //
        // We have to build the CSR structure manually since Eigen doesn't
        // expose a "take ownership of these buffers" API.
        //
        // Strategy: read into mat's internal storage via valuePtr/innerIndexPtr.
        mat.resizeNonZeros(nnz);

        H5Handle ds_vals{ H5Dopen2(grp.id, "data",    H5P_DEFAULT), H5Dclose };
        H5Handle ds_cols{ H5Dopen2(grp.id, "indices", H5P_DEFAULT), H5Dclose };

        // Read values and col indices directly into Eigen's buffers
        H5Dread(ds_vals.id, h5Type<Scalar>(),  H5S_ALL, H5S_ALL,
                H5P_DEFAULT, mat.valuePtr());
        H5Dread(ds_cols.id, H5T_NATIVE_INT32,  H5S_ALL, H5S_ALL,
                H5P_DEFAULT, mat.innerIndexPtr());

        // Fill outer index (row pointers), narrowing int64→int32
        for (int i = 0; i <= nrows; ++i)
            mat.outerIndexPtr()[i] = static_cast<int>(csr_rowptr[i]);

        mat.makeCompressed();
        return mat;
    }

    // ── Dense load — correct row-major → column-major conversion ────
    template <typename Scalar>
    DenseMatrix<Scalar> loadDense(const std::string& path) const {
        H5Handle ds{ H5Dopen2(file_, path.c_str(), H5P_DEFAULT), H5Dclose };
        if (ds.id < 0)
            throw std::runtime_error("Failed to open dataset: " + path);

        // Validate dimensionality
        H5Handle space{ H5Dget_space(ds.id), H5Sclose };
        int ndims = H5Sget_simple_extent_ndims(space.id);
        if (ndims != 2)
            throw std::runtime_error("Dataset '" + path + "' is not 2-D (ndims=" +
                                     std::to_string(ndims) + ").");

        hsize_t dims[2] = {0, 0};
        H5Sget_simple_extent_dims(space.id, dims, nullptr);

        const int nrows = static_cast<int>(dims[0]);
        const int ncols = static_cast<int>(dims[1]);

        // Check on-disk dtype matches requested Scalar to avoid silent garbage
        {
            H5Handle file_type{ H5Dget_type(ds.id),   H5Tclose };
            H5Handle native    { H5Tget_native_type(file_type.id, H5T_DIR_ASCEND), H5Tclose };
            if (H5Tget_size(native.id) != sizeof(Scalar))
                throw std::runtime_error(
                    "Dataset '" + path + "': on-disk element size (" +
                    std::to_string(H5Tget_size(native.id)) +
                    " bytes) does not match requested Scalar (" +
                    std::to_string(sizeof(Scalar)) + " bytes). "
                    "Use the correct template parameter.");
        }

        // HDF5 stores row-major. Read into a row-major buffer, then copy
        // into Eigen's column-major DenseMatrix correctly.
        std::vector<Scalar> buf(static_cast<size_t>(nrows) * ncols);
        H5Dread(ds.id, h5Type<Scalar>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());

        using RowMajorMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<const RowMajorMatrix> view(buf.data(), nrows, ncols);
        return DenseMatrix<Scalar>(view);   // explicit conversion — correct layout
    }

    // ── Auto-detecting load ──────────────────────────────────────────
    template <typename Scalar>
    MatrixVariant<Scalar> load(const std::string& path) const {
        switch (objectType(path)) {
            case H5I_GROUP:   return loadSparse<Scalar>(path);
            case H5I_DATASET: return loadDense<Scalar>(path);
            default:
                throw std::runtime_error(
                    "Path '" + path + "' is neither a group nor a dataset.");
        }
    }

    // ── Ground-truth helper — loads and converts to 0-based ─────────
    // scipy saves knns as 1-based; this adjusts automatically.
    Eigen::MatrixXi loadGroundTruth(const std::string& path,
                                    bool one_based = true) const {
        auto gt = loadDense<int>(path);
        if (one_based) gt.array() -= 1;
        return gt;
    }

private:
    hid_t file_ = -1;

    H5I_type_t objectType(const std::string& path) const {
        H5Handle obj{ H5Oopen(file_, path.c_str(), H5P_DEFAULT), H5Oclose };
        if (obj.id < 0)
            throw std::runtime_error("Path not found in HDF5 file: " + path);
        return H5Iget_type(obj.id);
    }

    static std::pair<int, int> readShapeAttr(hid_t grp,
                                              const std::string& path) {
        H5Handle attr{ H5Aopen(grp, "shape", H5P_DEFAULT), H5Aclose };
        if (attr.id < 0)
            throw std::runtime_error("Group '" + path +
                                     "' missing 'shape' attribute.");
        int64_t shape[2] = {0, 0};
        H5Aread(attr.id, H5T_NATIVE_INT64, shape);
        return { static_cast<int>(shape[0]), static_cast<int>(shape[1]) };
    }

    template <typename T>
    static std::vector<T> read1D(hid_t grp, const std::string& name) {
        H5Handle ds{ H5Dopen2(grp, name.c_str(), H5P_DEFAULT), H5Dclose };
        if (ds.id < 0)
            throw std::runtime_error("Failed to open dataset: " + name);

        H5Handle space{ H5Dget_space(ds.id), H5Sclose };
        hsize_t len = 0;
        H5Sget_simple_extent_dims(space.id, &len, nullptr);

        std::vector<T> buf(len);
        H5Dread(ds.id, h5Type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
        return buf;
    }

    template <typename T>
    static hid_t h5Type() {
        if constexpr (std::is_same_v<T, float>)    return H5T_NATIVE_FLOAT;
        if constexpr (std::is_same_v<T, double>)   return H5T_NATIVE_DOUBLE;
        if constexpr (std::is_same_v<T, int32_t>)  return H5T_NATIVE_INT32;
        if constexpr (std::is_same_v<T, int64_t>)  return H5T_NATIVE_INT64;
        if constexpr (std::is_same_v<T, uint32_t>) return H5T_NATIVE_UINT32;
        throw std::runtime_error("Unsupported scalar type for HDF5 read.");
    }
};