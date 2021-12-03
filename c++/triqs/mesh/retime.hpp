// Copyright (c) 2014-2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2014-2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018-2020 Simons Foundation
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You may obtain a copy of the License at
//     https://www.gnu.org/licenses/gpl-3.0.txt
//
// Authors: Michel Ferrero, Olivier Parcollet, Nils Wentzell

#pragma once
#include "triqs/mesh/domains/real_complex.hpp"

namespace triqs::mesh {

  struct retime : linear_mesh<real_domain> {

    retime() = default;

    /**
     *  Construct a Mesh of real times
     *
     *  Times are evenly distributed in the interval [t_min, t_max]
     *  including values at both edges.
     *
     *  @param t_min Smallest time
     *  @param t_max Largest time
     *  @param n_t Number of time-points
     */
    retime(double x_min, double x_max, int n_t) : linear_mesh<real_domain>(real_domain{}, x_min, x_max, n_t) {}
    template <typename... T> retime(T &&... x) : linear_mesh<real_domain>(std::forward<T>(x)...) {}

    retime(std::pair<double, double> window, int n_t) : retime(std::get<0>(window), std::get<1>(window), n_t) {}

    friend std::ostream &operator<<(std::ostream &sout, retime const &m) {
      return sout << "Real Time Mesh of size " << m.size() << ", t_min: " << m.t_min() << ", t_max: " << m.t_max();
    }

    // Smallest frequency in the mesh
    double t_min() const { return x_min(); }

    // Largest frequency in the mesh
    double t_max() const { return x_max(); }

    // -------------------- HDF5 -------------------

    static std::string hdf5_format() { return "MeshReTime"; }

    friend void h5_write(h5::group fg, std::string const &subgroup_name, retime const &m) { h5_write_impl(fg, subgroup_name, m, "MeshReTime"); }

    friend void h5_read(h5::group fg, std::string const &subgroup_name, retime &m) { h5_read_impl(fg, subgroup_name, m, "MeshReTime"); }
  };
} // namespace triqs::mesh