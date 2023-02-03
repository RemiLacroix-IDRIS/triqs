
namespace triqs::mesh {

  /*----------------------------------------------------------
  *  closest_point mechanism
  *  This trait will contain the specialisation to get
  *  the closest point ...
  *--------------------------------------------------------*/
  template <typename Mesh, typename Target> struct closest_point;

  // implementation
  template <typename... T> struct closest_pt_wrap;

  template <typename T> struct closest_pt_wrap<T> : tag::mesh_point {
    T value;
    template <typename U> explicit closest_pt_wrap(U &&x) : value(std::forward<U>(x)) {}
  };

  template <typename T1, typename T2, typename... Ts> struct closest_pt_wrap<T1, T2, Ts...> : tag::mesh_point {
    std::tuple<T1, T2, Ts...> value_tuple;
    template <typename... U> explicit closest_pt_wrap(U &&... x) : value_tuple(std::forward<U>(x)...) {}
  };

  /// DOC ?
  template <typename... T> closest_pt_wrap<T...> closest_mesh_pt(T &&... x) { return closest_pt_wrap<T...>{std::forward<T>(x)...}; }

  //-------------------------------------------------------
  // imtime
  // ------------------------------------------------------

  template <typename Target> struct closest_point<imtime, Target> {
    template <typename M, typename T> static int invoke(M const &mesh, closest_pt_wrap<T> const &p) {
      EXPECTS(mesh.x_min() < mesh.x_max());
      EXPECTS(mesh.x_min() <= p.value);
      EXPECTS(p.value <= mesh.x_max());
      return static_cast<int>(p.value * mesh.delta_inv() + 0.5);
    }
  };

  //-------------------------------------------------------
  // linear mesh (refreq ...), cf below
  // ------------------------------------------------------

  struct closest_point_linear_mesh {
    template <typename M, typename T> static int invoke(M const &mesh, closest_pt_wrap<T> const &p) {
      EXPECTS(mesh.x_min() < mesh.x_max());
      EXPECTS(mesh.x_min() <= p.value);
      EXPECTS(p.value <= mesh.x_max());
      return static_cast<int>((p.value - mesh.x_min()) * mesh.delta_inv() + 0.5);
    }
  };

  // For all mesh represented by a linear grid, the code is the same
  template <typename Target> struct closest_point<retime, Target> : closest_point_linear_mesh {};
  template <typename Target> struct closest_point<refreq, Target> : closest_point_linear_mesh {};

  //-------------------------------------------------------
  // brzone
  // ------------------------------------------------------

  template <typename Target> struct closest_point<brzone, Target> {

    template <typename T> static brzone::index_t invoke(brzone const &m, closest_pt_wrap<T> const &p) {

      // calculate k in the brzone basis
      auto k_units = transpose(inverse(m.units())) * nda::vector_const_view<double>{{3}, p.value.data()};
      long n1      = std::floor(k_units[0]);
      long n2      = std::floor(k_units[1]);
      long n3      = std::floor(k_units[2]);

      // calculate position relative to neighbors in mesh
      auto w1 = k_units[0] - n1;
      auto w2 = k_units[1] - n2;
      auto w3 = k_units[2] - n3;

      // fold back to brzone mesh
      auto n_folded = m.index_modulo(std::array{n1, n2, n3});

      // prepare result container and distance measure
      brzone::index_t res;
      auto dst = std::numeric_limits<double>::infinity();

      // find nearest neighbor by comparing distances
      // TODO: dimension check for speedup if dim < 3
      for (auto const &[i1, i2, i3] : itertools::product_range(2, 2, 2)) {
        std::array<double, 3> dst_vec = {w1 - i1, w2 - i2, w3 - i3};
        auto dst_vecp                 = transpose(m.units()) * nda::vector_const_view<double>{{3}, dst_vec.data()};
        auto dstp                     = nda::blas::dot(dst_vecp, dst_vecp);

        // update result when distance is smaller than current
        if (dstp < dst) {
          dst    = dstp;
          res[0] = n_folded[0] + i1;
          res[1] = n_folded[1] + i2;
          res[2] = n_folded[2] + i3;
        }
      }

      // fold back to brzone mesh (nearest neighbor could be out of bounds)
      return m.index_modulo(res);
    }
  };

  //-------------------------------------------------------
  // closest mesh point on the grid
  // ------------------------------------------------------

  template <typename... Ms, typename Target> struct closest_point<prod<Ms...>, Target> {
    using index_t = typename prod<Ms...>::index_t;

    template <typename M, typename... T, size_t... Is> static index_t _impl(M const &m, closest_pt_wrap<T...> const &p, std::index_sequence<Is...>) {
      return index_t(closest_point<Ms, Target>::invoke(std::get<Is>(m), closest_pt_wrap<T>{std::get<Is>(p.value_tuple)})...);
    }

    template <typename M, typename... T> static index_t invoke(M const &m, closest_pt_wrap<T...> const &p) {
      return _impl(m, p, std::index_sequence_for<T...>{});
    }
  };

} // namespace triqs::mesh
