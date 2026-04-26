#pragma once

#include <string>
#include <vector>
#include <stdexcept>

#include <hdf5.h>
#include <Eigen/Sparse>

// Loads a sparse matrix from an HDF5 file.
//
// Expected HDF5 layout (scipy.sparse / h5py convention):
//   <group>/data            – non-zero values
//   <group>/indices         – column indices (int32)
//   <group>/indptr          – row pointers   (int32)
//   <group>.attrs["shape"]  – [nrows, ncols]
//
// Usage:
//   HDF5SparseLoader loader("nq.h5");
//   auto mat = loader.load<float>("train");

class HDF5SparseLoader {
public:
    explicit HDF5SparseLoader(const std::string& filepath) {
        file = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0)
            throw std::runtime_error("Failed to open HDF5 file: " + filepath);
    }

    ~HDF5SparseLoader() {
        if (file >= 0) H5Fclose(file);
    }

    HDF5SparseLoader(const HDF5SparseLoader&)            = delete;
    HDF5SparseLoader& operator=(const HDF5SparseLoader&) = delete;

    template <typename Scalar>
    Eigen::SparseMatrix<Scalar, Eigen::RowMajor> load(const std::string& group) const {
        hid_t grp = H5Gopen2(file, group.c_str(), H5P_DEFAULT);
        if (grp < 0)
            throw std::runtime_error("Failed to open group: " + group);

        auto [nrows, ncols] = readShapeAttr(grp, group);

        auto values  = read1D<Scalar>(grp, "data");
        auto indices = read1D<int>   (grp, "indices");
        auto indptr  = read1D<int>   (grp, "indptr");
        H5Gclose(grp);

        // Map the raw CSR arrays directly into Eigen — no element-wise loop.
        Eigen::Map<const Eigen::SparseMatrix<Scalar, Eigen::RowMajor>> mapped(
            nrows,
            ncols,
            static_cast<int>(values.size()),
            indptr.data(),
            indices.data(),
            values.data()
        );

        // Explicit copy to get an owning matrix (the vectors above will go out of scope).
        return Eigen::SparseMatrix<Scalar, Eigen::RowMajor>(mapped);
    }

private:
    hid_t file = -1;

    static std::pair<int, int> readShapeAttr(hid_t grp, const std::string& group) {
        hid_t attr = H5Aopen(grp, "shape", H5P_DEFAULT);
        if (attr < 0)
            throw std::runtime_error("Group '" + group + "' missing 'shape' attribute.");
        hsize_t shape[2] = {0, 0};
        H5Aread(attr, H5T_NATIVE_HSIZE, shape);
        H5Aclose(attr);
        return {static_cast<int>(shape[0]), static_cast<int>(shape[1])};
    }

    template <typename T>
    static std::vector<T> read1D(hid_t grp, const std::string& name) {
        hid_t ds = H5Dopen2(grp, name.c_str(), H5P_DEFAULT);
        if (ds < 0)
            throw std::runtime_error("Failed to open dataset: " + name);

        hid_t space = H5Dget_space(ds);
        hsize_t len = 0;
        H5Sget_simple_extent_dims(space, &len, nullptr);
        H5Sclose(space);

        std::vector<T> buf(len);
        H5Dread(ds, h5Type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
        H5Dclose(ds);
        return buf;
    }

    template <typename T>
    static hid_t h5Type() {
        if constexpr (std::is_same_v<T, float>)  return H5T_NATIVE_FLOAT;
        if constexpr (std::is_same_v<T, double>) return H5T_NATIVE_DOUBLE;
        if constexpr (std::is_same_v<T, int>)    return H5T_NATIVE_INT32;
        throw std::runtime_error("Unsupported scalar type.");
    }
};
