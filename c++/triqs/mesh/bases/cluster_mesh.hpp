/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2014 by O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once

#include <nda/nda.hpp>
#include <utility>

#include "../details/mesh_tools.hpp"
#include "../domains/lattice.hpp"
#include "../details/mesh_point.hpp"

namespace triqs::mesh {

  class cluster_mesh {
    public:
    ///type of the domain: matsubara_freq_domain
    using domain_t = lattice_domain;

    ///type of the Lattice index
    using index_t = std::array<long, 3>;

    ///type of the linear index
    using linear_index_t = long;

    ///type of the domain point
    using domain_pt_t = domain_t::point_t;

    // -------------------- Constructors -------------------

    ///
    cluster_mesh() = default;

    /**
     * Construct from lattice_domain and dims
     *
     * @param dom The underlying lattice domain
     * @param dims 1D Array of rank 3 specifying the periodicity for each dimension
     */
    cluster_mesh(lattice_domain dom, std::array<long, 3> dims)
       : dom_(std::move(dom)),
         dims_(dims),                           //
         size_(dims_[0] * dims_[1] * dims_[2]), //
         stride0{dims_[1] * dims_[2]},          //
         stride1{dims_[2]} {}

    /**
     * Construct from basis vectors and dims
     *
     * @param units Matrix $B$ containing as rows the basis vectors that generate mesh points
     * @param dims 1D Array of rank 3 specifying the periodicity for each dimension
     */
    cluster_mesh(matrix<double> units, std::array<long, 3> dims) : cluster_mesh(lattice_domain(std::move(units)), dims) {}

    // -------------------- Comparisons -------------------

    bool operator==(cluster_mesh const &M) const { return ((dims_ == M.dims_) && (units() == M.units())); }
    bool operator!=(cluster_mesh const &M) const = default;

    // -------------------- Index Modulo -------------------

    /// Modulo operation on a given index, applying component-wise modulo-operations using dims
    index_t index_modulo(index_t const &idx) const {
      static auto _modulo = [&](long r, int i) {
        long res = r % dims_[i];
        return (res >= 0 ? res : res + dims_[i]);
      };
      return index_t{_modulo(idx[0], 0), _modulo(idx[1], 1), _modulo(idx[2], 2)};
    }

    // -------------------- Mesh Concept -------------------

    [[nodiscard]] size_t mesh_hash() const { return mesh_hash_; }

    /// The underlying domain
    domain_t const &domain() const { return dom_; }

    /// The total number of points in the mesh
    size_t size() const { return size_; }

    /// Is the point in the mesh ? Always true
    template <typename T> static constexpr bool is_within_boundary(lattice_point const &) { return true; }

    /// From an index of a point in the mesh, returns the corresponding point in the domain
    [[nodiscard]] domain_pt_t index_to_point(index_t const &idx) const {
      EXPECTS(idx == index_modulo(n));
      return domain_pt_t{idx, dom_.units};
    }

    /// Flatten the index
    [[nodiscard]] linear_index_t index_to_linear(index_t const &idx) const {
      EXPECTS(idx == index_modulo(idx));
      return idx[0] * stride0 + idx[1] * stride1 + idx[2];
    }

    /// Unflatten index
    [[nodiscard]] index_t linear_to_index(linear_index_t const &lidx) const {
      long i0 = lidx / stride0;
      long r0 = lidx % stride0;
      long i1 = r0 / stride1;
      long i2 = i0 % stride1;
      return {i0, i1, i2};
    }

    // -------------------- Accessors (other) -------------------

    int rank() const { return (dims_[2] > 1 ? 3 : (dims_[1] > 1 ? 2 : 1)); }

    /// The extent of each dimension
    std::array<long, 3> dims() const { return dims_; }

    /// Matrix containing the mesh basis vectors as rows
    matrix_const_view<double> units() const { return dom_.units; }

    // -------------------------- Other --------------------------

    ///// locate the closest point
    //inline index_t closest_index(domain_pt_t const &x) const {
    //auto idbl = transpose(inverse(dom_.units)) * x;
    //return {std::lround(idbl[0]), std::lround(idbl[1]), std::lround(idbl[2])};
    //}

    // -------------------- mesh_point -------------------

    struct mesh_point_t : public lattice_point {
      using mesh_t = cluster_mesh;

      mesh_point_t(mesh_t::index_t const &idx, matrix<double> const &units, mesh_t::linear_index_t linear_index, size_t mesh_hash)
         : lattice_point(idx, units), linear_index_(linear_index), mesh_hash_(mesh_hash) {}

      [[nodiscard]] auto index() const { return idx; }
      [[nodiscard]] auto value() const { return cast_t(*this); }
      [[nodiscard]] auto linear_index() const { return linear_index_; }
      [[nodiscard]] auto mesh_hash() const { return mesh_hash_; }

      // The mesh point behaves like a vector
      double operator()(int d) const { return value()[d]; }
      double operator[](int d) const { return value()[d]; }

      friend std::ostream &operator<<(std::ostream &out, mesh_point_t const &x) { return out << (lattice_point)x; }

      private:
      cluster_mesh::linear_index_t linear_index_{};
      std::size_t mesh_hash_ = 0;
    };

    /// Accessing a point of the mesh from its index
    [[nodiscard]] mesh_point_t operator[](index_t idx) const {
      EXPECTS(idx == index_modulo(idx)); // TODO: Is this requirement correct?
      return {idx, dom_.units, index_to_linear(idx), mesh_hash_};
    }
    [[nodiscard]] mesh_point_t linear_to_mesh_pt(linear_index_t const &linear_index) const {
      return {linear_to_index(linear_index), dom_.units, linear_index, mesh_hash_};
    }

    // -------------------------- Range & Iteration --------------------------

    [[nodiscard]] auto begin() const {
      r_ = make_mesh_range(*this);
      return r_.begin();
    }
    [[nodiscard]] auto end() const { return r_.end(); }
    [[nodiscard]] auto cbegin() const {
      r_ = make_mesh_range(*this);
      return r_.begin();
    }
    [[nodiscard]] auto cend() const { return r_.end(); }

    // -------------- HDF5  --------------------------

    /// Write into HDF5
    friend void h5_write_impl(h5::group fg, std::string subgroup_name, cluster_mesh const &m) {
      h5::group gr = fg.create_group(subgroup_name);
      write_hdf5_format(gr, m);
      h5_write(gr, "domain", m.dom_);
      h5_write(gr, "dims", m.dims_);
    }

    /// Read from HDF5
    friend void h5_read_impl(h5::group fg, std::string subgroup_name, cluster_mesh &m) {
      h5::group gr = fg.open_group(subgroup_name);
      assert_hdf5_format(gr, m, true);
      auto dom  = h5_read<cluster_mesh::domain_t>(gr, "domain");
      auto dims = h5_read<std::array<long, 3>>(gr, "dims");
      m         = cluster_mesh(dom, dims);
    }

    // -------------------- print  -------------------

    friend std::ostream &operator<<(std::ostream &sout, cluster_mesh const &m) {
      return sout << "cluster_mesh of size " << m.dims() << "\n units = " << m.units() << "\n dims = " << m.dims()
                  << "\n";
    }

    // ------------------------------------------------

    private:
    domain_t dom_;
    std::array<long, 3> dims_;
    size_t size_;
    long stride1, stride0;
    size_t mesh_hash_ = 0;
    mutable make_mesh_range_rtype<cluster_mesh> r_;

    public:
    using const_iterator = decltype(r_.begin());
    using iterator       = const_iterator;
  };

} // namespace triqs::mesh
