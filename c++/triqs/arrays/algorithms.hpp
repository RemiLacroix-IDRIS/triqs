// Copyright (c) 2013-2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2013-2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018-2019 Simons Foundation
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
// Authors: Olivier Parcollet, Nils Wentzell

#pragma once
#include "./functional/fold.hpp"
#include <algorithm>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <functional>
#include <triqs/utility/first_include.hpp>

namespace triqs {
  namespace arrays {

// get_first_element(a) returns .. the first element, i.e. a(0,0,0...)
#define ZERO(z, n, text) 0
#define IMPL(z, N, unused)                                                                                                                           \
  template <typename ArrayType>                                                                                                                      \
  typename std::enable_if<ArrayType::domain_type::rank == N, typename ArrayType::value_type>::type get_first_element(ArrayType const &A) {           \
    return A(BOOST_PP_ENUM(N, ZERO, 0));                                                                                                             \
  }
    BOOST_PP_REPEAT_FROM_TO(1, ARRAY_NRANK_MAX, IMPL, nil);
#undef IMPL
#undef ZERO

    inline double max_element(double x) { return x; }
    inline std::complex<double> max_element(std::complex<double> x) { return x; }

    template <class A> std::enable_if_t<ImmutableCuboidArray<A>::value, typename A::value_type> max_element(A const &a) {
      return fold([](auto const &a, auto const &b) { return std::max(a, b); })(a, get_first_element(a));
    }

    template <class A> std::enable_if_t<ImmutableCuboidArray<A>::value, typename A::value_type> min_element(A const &a) {
      return fold([](auto const &a, auto const &b) { return std::min(a, b); })(a, get_first_element(a));
    }

    template <class A> std::enable_if_t<ImmutableCuboidArray<A>::value, typename A::value_type> sum(A const &a) {
      return fold(std::plus<typename A::value_type>())(a);
    }

  } // namespace arrays
} // namespace triqs
