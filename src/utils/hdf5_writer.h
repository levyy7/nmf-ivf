#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <functional>

#include <hdf5.h>
#include <Eigen/Dense>

class HDF5Writer {
public:
  struct H5Handle {
    hid_t id = -1;
    std::function<herr_t(hid_t)> closer;

    H5Handle(hid_t id, std::function<herr_t(hid_t)> closer)
      : id(id), closer(std::move(closer)) {
    }

    ~H5Handle() { if (id >= 0) closer(id); }

    H5Handle(const H5Handle&) = delete;
    H5Handle& operator=(const H5Handle&) = delete;

    operator hid_t() const { return id; }
  };

  explicit HDF5Writer(const std::string& filepath) {
    // Temporarily disable HDF5 error printing to silently check if file exists
    H5E_auto2_t old_func;
    void* old_client_data;
    H5Eget_auto2(H5E_DEFAULT, &old_func, &old_client_data);
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    // Try opening in Read/Write mode first
    file_ = H5Fopen(filepath.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

    // Restore error printing
    H5Eset_auto2(H5E_DEFAULT, old_func, old_client_data);

    // If it doesn't exist, create it
    if (file_ < 0) {
      file_ = H5Fcreate(filepath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                        H5P_DEFAULT);
      if (file_ < 0) {
        throw std::runtime_error(
            "Failed to open or create HDF5 file: " + filepath);
      }
    }
  }

  ~HDF5Writer() { if (file_ >= 0) H5Fclose(file_); }

  HDF5Writer(const HDF5Writer&) = delete;
  HDF5Writer& operator=(const HDF5Writer&) = delete;

  void writeKnns(const Eigen::MatrixXi& knns, bool is_zero_based = true) {
    if (is_zero_based) {
      Eigen::MatrixXi knns_1based = knns.array() + 1;
      // Padded -1s will safely become 0s
      writeDenseDataset(file_, "knns", knns_1based);
    } else {
      writeDenseDataset(file_, "knns", knns);
    }
  }

  void writeDists(const Eigen::MatrixXf& dists) {
    writeDenseDataset(file_, "dists", dists);
  }

  void writeIndex(const Eigen::MatrixXf& H, const Eigen::MatrixXi& lists) {
    // Delete the entire group if it exists so we can safely overwrite it
    deleteIfExists(file_, "index");

    H5Handle grp{
        H5Gcreate2(file_, "index", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
        H5Gclose};
    if (grp.id < 0) {
      throw std::runtime_error("Failed to create group: index");
    }

    writeDenseDataset(grp.id, "H", H);
    writeDenseDataset(grp.id, "lists", lists);
  }

  void writeAttribute(const std::string& attr_name, const std::string& value) {
    hid_t root = H5Gopen2(file_, "/", H5P_DEFAULT);
    if (root < 0) return;

    if (H5Aexists(root, attr_name.c_str()) > 0) {
      H5Adelete(root, attr_name.c_str());
    }

    hid_t type = H5Tcopy(H5T_C_S1);
    H5Tset_size(type, value.empty() ? 1 : value.size() + 1);
    H5Tset_strpad(type, H5T_STR_NULLTERM);

    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(root, attr_name.c_str(), type, space, H5P_DEFAULT,
                            H5P_DEFAULT);
    if (attr >= 0) {
      H5Awrite(attr, type, value.c_str());
      H5Aclose(attr);
    }
    H5Sclose(space);
    H5Tclose(type);
    H5Gclose(root);
  }

  void writeAttribute(const std::string& attr_name, float value) {
    hid_t root = H5Gopen2(file_, "/", H5P_DEFAULT);
    if (root < 0) return;

    if (H5Aexists(root, attr_name.c_str()) > 0) {
      H5Adelete(root, attr_name.c_str());
    }

    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(root, attr_name.c_str(), H5T_NATIVE_FLOAT, space,
                            H5P_DEFAULT, H5P_DEFAULT);
    if (attr >= 0) {
      H5Awrite(attr, H5T_NATIVE_FLOAT, &value);
      H5Aclose(attr);
    }
    H5Sclose(space);
    H5Gclose(root);
  }

private:
  hid_t file_ = -1;

  // Helper to unlink (delete) a dataset or group if it already exists
  void deleteIfExists(hid_t loc_id, const std::string& name) {
    htri_t exists = H5Lexists(loc_id, name.c_str(), H5P_DEFAULT);
    if (exists > 0) {
      H5Ldelete(loc_id, name.c_str(), H5P_DEFAULT);
    }
  }

  template <typename Derived>
  void writeDenseDataset(hid_t loc_id, const std::string& name,
                         const Eigen::MatrixBase<Derived>& mat) {
    using Scalar = typename Derived::Scalar;

    // Ensure we are overwriting cleanly
    deleteIfExists(loc_id, name);

    // Convert column-major Eigen to row-major for disk
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        row_major_mat = mat;

    hsize_t dims[2] = {
        static_cast<hsize_t>(mat.rows()),
        static_cast<hsize_t>(mat.cols())
    };

    H5Handle space{H5Screate_simple(2, dims, nullptr), H5Sclose};
    if (space.id < 0)
      throw std::runtime_error(
          "Failed to create dataspace for " + name);

    H5Handle ds{H5Dcreate2(loc_id, name.c_str(), h5Type<Scalar>(), space.id,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                H5Dclose};
    if (ds.id < 0) throw std::runtime_error("Failed to create dataset " + name);

    herr_t status = H5Dwrite(ds.id, h5Type<Scalar>(), H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, row_major_mat.data());
    if (status < 0)
      throw std::runtime_error(
          "Failed to write data to dataset " + name);
  }

  template <typename T>
  static hid_t h5Type() {
    if constexpr (std::is_same_v<T, float>) return H5T_NATIVE_FLOAT;
    if constexpr (std::is_same_v<T, double>) return H5T_NATIVE_DOUBLE;
    if constexpr (std::is_same_v<T, int32_t>) return H5T_NATIVE_INT32;
    if constexpr (std::is_same_v<T, int64_t>) return H5T_NATIVE_INT64;
    if constexpr (std::is_same_v<T, uint32_t>) return H5T_NATIVE_UINT32;
    throw std::runtime_error("Unsupported scalar type for HDF5 write.");
  }
};