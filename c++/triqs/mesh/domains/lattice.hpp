#pragma once

#include <fmt/core.h>
#include <nda/h5.hpp>

#include <triqs/utility/exceptions.hpp>
#include "../details/mesh_tools.hpp"

namespace triqs::mesh {

  struct lattice_point {
    std::array<long, 3> idx = {0,0,0};
    nda::matrix<double> const &units = nda::eye<double>(3);

    lattice_point() = default;
    lattice_point(std::array<long, 3> const &idx_, matrix<double> const &units_) : idx(idx_), units(units_) {}

    using cast_t = nda::vector<double>;
    operator cast_t() const {
      cast_t M(3);
      M() = 0.0;
      // slow, to be replaced with matrix vector multiplication
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) M(i) += idx[j] * units(j, i);
      return M;
    }
    operator std::array<long, 3>() const {
      return idx;
    }

    friend std::ostream &operator<<(std::ostream &out, lattice_point const &x) { return out << (cast_t)x; }

  };

  inline lattice_point operator+(lattice_point const &x, lattice_point const &y) {
    EXPECTS(x.units == y.units);
    return {x.idx + y.idx, x.units};
  }

  inline lattice_point operator-(lattice_point const &x, lattice_point const &y) {
    EXPECTS(x.units == y.units);
    return {x.idx - y.idx, x.units};
  }

  inline lattice_point operator-(lattice_point const &x) { return {-x.idx, x.units}; }

  //---------------------------------------------------------------------------------------------------------

  class bravais_lattice {
    public:
    using point_t = lattice_point;

    [[nodiscard]] bool contains(point_t const &pt) const { return true; }

    bravais_lattice() = default;
    bravais_lattice(matrix<double> units) : units_{std::move(units)} {
      if (nda::determinant(units_) == 0) TRIQS_RUNTIME_ERROR << "Unit vectors must not be collinear";
    }

    /// Number of dimensions
    int ndim() const { return 3; }

    /// Matrix containing lattice basis vectors as rows
    nda::matrix_const_view<double> units() const { return units_; }

    bool operator==(bravais_lattice const &) const = default;
    bool operator!=(bravais_lattice const &) const = default;

    static std::string hdf5_format() { return "bravais_lattice"; }

    /// Write into HDF5
    friend void h5_write(h5::group fg, std::string const &subgroup_name, bravais_lattice const &d) {
      h5::group gr = fg.create_group(subgroup_name);
      h5_write(gr, "units", d.units);
    }

    /// Read from HDF5
    friend void h5_read(h5::group fg, std::string const &subgroup_name, bravais_lattice &d) {
      h5::group gr          = fg.open_group(subgroup_name);
      h5_read(gr, "units", d.units);
    }

    friend std::ostream &operator<<(std::ostream &sout, bravais_lattice const &d) {
      return sout << fmt::format("Bravais Lattice with unit vectors = {}", d.units);
    }

    private:
    std::vector<std::array<double, 3>> atom_orb_pos = {{0,0,0}};
    std::vector<std::array<double, 3>> atom_orb_name = {{0,0,0}};
    nda::matrix<double> units_ = nda::eye<double>(3);
  };

} // namespace triqs::mesh
