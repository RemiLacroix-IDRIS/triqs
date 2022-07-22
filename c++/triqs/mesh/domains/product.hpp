// Copyright (c) 2013-2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2013-2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018 Simons Foundation
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
#include "triqs/utility/tuple_tools.hpp"
#include <h5/h5.hpp>
#include <tuple>

namespace triqs::mesh {

  template <typename... Domains> struct domain_product {
    using point_t = std::tuple<typename Domains::point_t...>;
    std::tuple<Domains...> domains;

    domain_product() = default;
    domain_product(std::tuple<Domains...> const &dom_tpl) : domains(dom_tpl) {}
    domain_product(std::tuple<Domains...> &&dom_tpl) : domains(std::move(dom_tpl)) {}
    domain_product(Domains const &...doms) requires(sizeof...(Domains) > 0) : domains(doms...) {}

    [[nodiscard]] bool contains(point_t const &pt) const {
      return triqs::tuple::fold([](auto &m, auto &arg, bool r) { return r && (m.is_within_boundary(arg)); }, domains, pt, true);
    }

    friend bool operator==(domain_product const &D1, domain_product const &D2) { return D1.domains == D2.domains; }

    static std::string hdf5_format() { return "DomainProduct"; }

    /// Write into HDF5
    friend void h5_write(h5::group fg, std::string const &subgroup_name, domain_product const &dp) {
      h5::group gr = fg.create_group(subgroup_name);
      write_hdf5_format(gr, dp);
      auto l = [gr](int N, auto const &d) { h5_write(gr, "DomainComponent" + std::to_string(N), d); };
      triqs::tuple::for_each_enumerate(dp.domains, l);
    }

    /// Read from HDF5
    friend void h5_read(h5::group fg, std::string const &subgroup_name, domain_product &dp) {
      h5::group gr = fg.open_group(subgroup_name);
      assert_hdf5_format(gr, dp, true);
      auto l = [gr](int N, auto &m) { h5_read(gr, "DomainComponent" + std::to_string(N), m); };
      triqs::tuple::for_each_enumerate(dp.domains, l);
    }
  };
} // namespace triqs::mesh
