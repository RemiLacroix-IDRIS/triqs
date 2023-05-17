// Copyright (c) 2023 Hugo U. R. Strand
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
// Authors: Hugo U. R. Strand

#include <triqs/test_tools/gfs.hpp>
#include <h5/serialization.hpp>

TEST(Gf, dlr_imtime) {

  double beta   = 2.0;
  double lambda = 42.0;
  double eps = 1e-12;
  auto mesh = gf_mesh<triqs::mesh::dlr_imtime>{beta, Fermion, lambda, eps};

  std::cout << mesh << "\n";
  std::cout << "Rank " << mesh.size() << "\n";
  
  EXPECT_EQ(mesh.lambda(), lambda);
  EXPECT_EQ(mesh.eps(), eps);
  
  for (const auto &tau : mesh ) {
    std::cout << tau << "\n";
    EXPECT_TRUE(tau <= beta);
    EXPECT_TRUE(tau >= 0.0);
  }
}

TEST(Gf, dlr_imfreq_fermi) {

  double beta   = 2.0;
  double lambda = 1000.0;
  double eps = 1e-12;
  auto mesh = gf_mesh<triqs::mesh::dlr_imfreq>{beta, Fermion, lambda, eps};

  std::cout << mesh << "\n";
  std::cout << "Rank " << mesh.size() << "\n";
  
  EXPECT_EQ(mesh.lambda(), lambda);
  EXPECT_EQ(mesh.eps(), eps);
  
  for (const auto &iw : mesh ) {
    std::cout << iw.linear_index() << ", " << iw << "\n";
  }
}

TEST(Gf, dlr_imfreq_bose) {

  double beta   = 2.0;
  double lambda = 1000.0;
  double eps = 1e-12;
  auto mesh = gf_mesh<triqs::mesh::dlr_imfreq>{beta, Boson, lambda, eps};

  std::cout << mesh << "\n";
  std::cout << "Rank " << mesh.size() << "\n";
  
  EXPECT_EQ(mesh.lambda(), lambda);
  EXPECT_EQ(mesh.eps(), eps);
  
  for (const auto &iw : mesh ) {
    std::cout << iw.linear_index() << ", " << iw << "\n";
  }
}

TEST(Gf, dlr_coeffs) {

  // dlr_coeffs -> dlr_refreq
  
  double beta   = 2.0;
  double lambda = 42.0;
  double eps = 1e-12;
  auto mesh = gf_mesh<triqs::mesh::dlr_coeffs>{beta, Fermion, lambda, eps};

  std::cout << mesh << "\n";
  std::cout << "Rank " << mesh.size() << "\n";
  
  EXPECT_CLOSE(mesh.lambda(), lambda);
  EXPECT_CLOSE(mesh.eps(), eps);
  
  for (const auto &dlr_rf : mesh ) {
    std::cout << dlr_rf << "\n";
  }
}

TEST(Gf, dlr_coeffs_imtime) {

  double beta   = 2.0;
  double lambda = 42.0;
  double eps = 1e-12;
  
  auto cmesh = gf_mesh<triqs::mesh::dlr_coeffs>{beta, Fermion, lambda, eps};
  auto tmesh = gf_mesh<triqs::mesh::dlr_imtime>{beta, Fermion, lambda, eps};
  auto wmesh = gf_mesh<triqs::mesh::dlr_imfreq>{beta, Fermion, lambda, eps};

  std::cout << cmesh << "\n";
  std::cout << tmesh << "\n";
  std::cout << wmesh << "\n";

  auto cmesh_ref = triqs::mesh::dlr_coeffs(tmesh); // remove gf_mesh everywhere
  auto tmesh_ref = triqs::mesh::dlr_imtime(cmesh);
  auto wmesh_ref = triqs::mesh::dlr_imfreq(cmesh);
  
  for (const auto &[c1, c2] : itertools::zip(cmesh, cmesh_ref) ) {
    EXPECT_CLOSE(c1, c2);
  } 
  for (const auto &[t1, t2] : itertools::zip(tmesh, tmesh_ref) ) {
    EXPECT_CLOSE(t1, t2);
  }
  for (const auto &[w1, w2] : itertools::zip(wmesh, wmesh_ref) ) {
    EXPECT_CLOSE(w1, w2);
  }
}

TEST(Gf, dlr_imtime_grid) {

  double beta   = 2.0;
  double lambda = 42.0;
  double eps    = 1e-10;
  double omega  = 1.337;

  auto mesh = gf_mesh<triqs::mesh::dlr_imtime>{beta, Fermion, lambda, eps};
  auto G1 = gf<triqs::mesh::dlr_imtime, scalar_valued>{mesh};
  
  for (const auto &tau : mesh ) {
    G1[tau] = std::exp(-omega * tau) / (1 + std::exp(-beta * omega));
  }

  auto Gc = G1; // copy
  EXPECT_GF_NEAR(G1, Gc);

  auto Gv = G1(); // view
  EXPECT_GF_NEAR(Gv, Gc);
  
  // Take view and assign
  auto G2 = gf<triqs::mesh::dlr_imtime, scalar_valued>{mesh};
  auto G2v = gf_view(G2);
  for (const auto &tau : mesh ) {
    G2v[tau] = std::exp(-omega * tau) / (1 + std::exp(-beta * omega));
  }
  
  EXPECT_GF_NEAR(G2, G2v);
  EXPECT_GF_NEAR(G1, G2v);
  EXPECT_GF_NEAR(G1, G2);

  auto GpG = G1 + G2; // Addition
  for (const auto &tau : mesh ) {
    EXPECT_CLOSE(GpG[tau], G1[tau] * 2.);
    EXPECT_CLOSE(GpG[tau], 2. * G1[tau]);
  }

  auto G4 = 4. * G2; // Multiply with scalar
  for (const auto &tau : mesh ) {
    EXPECT_CLOSE(G4[tau], G1[tau] * 4.);
    EXPECT_CLOSE(G4[tau], 4. * G1[tau]);
  }

  auto GG = G1 * G2; // Multiplication
  for (const auto &tau : mesh ) {
    EXPECT_CLOSE(GG[tau], G1[tau] * G1[tau]);
  }

  std::cout << GG.mesh().domain() << "\n";
  std::cout << GG.mesh().domain().statistic << "\n";
  //EXPECT_TRUE(GG.mesh().domain().statistic == Boson); // Currently fails. FIXME!  
}

TEST(Gf, dlr_imfreq_grid) {

  double beta   = 2.0;
  double lambda = 42.0;
  double eps    = 1e-10;
  double omega  = 1.337;

  auto mesh = gf_mesh<triqs::mesh::dlr_imfreq>{beta, Fermion, lambda, eps};
  auto G1 = gf<triqs::mesh::dlr_imfreq, scalar_valued>{mesh};
  
  for (const auto &iw : mesh ) {
    G1[iw] = 1./(iw - omega);
  }

  auto Gc = G1; // copy
  EXPECT_GF_NEAR(G1, Gc);

  auto Gv = G1(); // view
  EXPECT_GF_NEAR(Gv, Gc);
  
  // Take view and assign
  auto G2 = gf<triqs::mesh::dlr_imfreq, scalar_valued>{mesh};
  auto G2v = gf_view(G2);
  for (const auto &iw : mesh ) {
    G2v[iw] = 1./(iw - omega);
  }
  
  EXPECT_GF_NEAR(G2, G2v);
  EXPECT_GF_NEAR(G1, G2v);
  EXPECT_GF_NEAR(G1, G2);

  auto GpG = G1 + G2; // Addition
  for (const auto &iw : mesh ) {
    EXPECT_CLOSE(GpG[iw], G1[iw] * 2.);
    EXPECT_CLOSE(GpG[iw], 2. * G1[iw]);
  }

  auto G4 = 4. * G2; // Multiply with scalar
  for (const auto &iw : mesh ) {
    EXPECT_CLOSE(G4[iw], G1[iw] * 4.);
    EXPECT_CLOSE(G4[iw], 4. * G1[iw]);
  }

  auto GG = G1 * G2; // Multiplication
  for (const auto &iw : mesh ) {
    EXPECT_CLOSE(GG[iw], G1[iw] * G1[iw]);
  }

  std::cout << GG.mesh().domain() << "\n";
  std::cout << GG.mesh().domain().statistic << "\n";
  //EXPECT_TRUE(GG.mesh().domain().statistic == Boson); // Currently fails. FIXME!  
}

TEST(Gf, dlr_imtime_grid_clef) {
  double beta   = 2.0;
  double lambda = 42.0;
  double eps    = 1e-10;
  double omega  = 1.337;

  auto G = gf<triqs::mesh::dlr_imtime, scalar_valued>{{beta, Fermion, lambda, eps}};
  triqs::clef::placeholder<0> tau_;
  G(tau_) << nda::clef::exp(-omega * tau_) / (1 + nda::clef::exp(-beta * omega));

  auto mesh = gf_mesh<triqs::mesh::dlr_imtime>{beta, Fermion, lambda, eps};
  auto G1 = gf<triqs::mesh::dlr_imtime, scalar_valued>{mesh};
  
  for (const auto &tau : mesh ) {
    G1[tau] = std::exp(-omega * tau) / (1 + std::exp(-beta * omega));
  }

  EXPECT_GF_NEAR(G, G1);
}

TEST(Gf, dlr_imfreq_grid_clef) {
  double beta   = 2.0;
  double lambda = 42.0;
  double eps    = 1e-10;
  double omega  = 1.337;

  auto G = gf<triqs::mesh::dlr_imfreq, scalar_valued>{{beta, Fermion, lambda, eps}};
  triqs::clef::placeholder<0> iw_;
  G(iw_) << 1. / (iw_ - omega);

  auto mesh = gf_mesh<triqs::mesh::dlr_imfreq>{beta, Fermion, lambda, eps};
  auto G1 = gf<triqs::mesh::dlr_imfreq, scalar_valued>{mesh};
  
  for (const auto &iw : mesh ) {
    G1[iw] = 1./(iw - omega);
  }

  EXPECT_GF_NEAR(G, G1);
}

TEST(Gf, dlr_imtime_interpolation) {

  double beta   = 2.0;
  double lambda = 42.0;
  double eps = 1e-12;

  long dlr_idx = 6; // Pick one DLR frequency

  auto G_dlr = gf<triqs::mesh::dlr_coeffs, scalar_valued>{{beta, Fermion, lambda, eps}};
  G_dlr() = 0.0;
  G_dlr.data()[dlr_idx] = 1.0;

  auto cmesh = G_dlr.mesh();
  auto tmesh = triqs::mesh::dlr_imtime(cmesh);
  
  auto G_tau = gf<triqs::mesh::dlr_imtime, scalar_valued>{tmesh};
  triqs::clef::placeholder<0> tau_;

  double omega = 1./beta * cmesh.index_to_point(dlr_idx);
  G_tau(tau_) << nda::clef::exp(-omega * tau_) / (1 + nda::clef::exp(-beta * omega));

  // Interpolation in imaginary time using dlr grid (efficient by design)

  for (auto const &tau : G_tau.mesh()) {
    EXPECT_CLOSE(G_tau[tau], G_dlr(tau));
  }
}

TEST(Gf, dlr_imfreq_interpolation) {

  double beta   = 2.0;
  double lambda = 42.0;
  double eps = 1e-12;
  double omega = 1.337;

  triqs::clef::placeholder<0> iw_;

  auto G_iw = gf<triqs::mesh::dlr_imfreq, scalar_valued>{{beta, Fermion, lambda, eps}};
  G_iw(iw_) << 1. / (iw_ + omega);

  auto G_dlr = dlr_coeffs_from_dlr_imfreq(G_iw);

  auto G_iw_ref = G_iw;
  G_iw_ref(iw_) << G_dlr(iw_); // Interpolate DLR in imaginary frequency

  for (auto const &iw : G_iw.mesh()) {
    EXPECT_CLOSE(G_iw[iw], G_iw_ref[iw]);
  }
  
  EXPECT_GF_NEAR(G_iw, G_iw_ref);
}

TEST(Gf, dlr_coeffs_conversion) {

  double beta   = 2.0;
  double lambda = 42.0;
  double eps = 1e-12;
  double omega = 1.337;

  triqs::clef::placeholder<0> tau_;

  auto G_tau = gf<triqs::mesh::dlr_imtime, scalar_valued>{{beta, Fermion, lambda, eps}};
  G_tau(tau_) << -nda::clef::exp(-omega * tau_) / (1 + nda::clef::exp(-beta * omega));

  // Transform from dlr_imtime to dlr_coeffs

  auto G_dlr = dlr_coeffs_from_dlr_imtime(G_tau);
  for (auto const &tau : G_tau.mesh()) EXPECT_CLOSE(G_tau[tau], G_dlr(tau));
  
  // Transform from dlr_coeffs to dlr_imtime

  auto G_tau_ref = dlr_imtime_from_dlr_coeffs(G_dlr);
  EXPECT_GF_NEAR(G_tau, G_tau_ref);

  // Transform from dlr_coeffs to dlr_imfreq and back

  auto G_iw = dlr_imfreq_from_dlr_coeffs(G_dlr);
  auto G_dlr_ref = dlr_coeffs_from_dlr_imfreq(G_iw);
  for (auto const &tau : G_tau.mesh()) EXPECT_CLOSE(G_tau[tau], G_dlr_ref(tau));

  auto G_iw_ref = G_iw;
  triqs::clef::placeholder<0> iw_;
  G_iw_ref(iw_) << 1. / (iw_ - omega);

  for (auto const &iw : G_iw.mesh()) {
    EXPECT_CLOSE(G_iw[iw], G_iw_ref[iw]);
  }

  EXPECT_GF_NEAR(G_iw, G_iw_ref);
}

TEST(Gf, dlr_density) {

  double beta   = 2.0;
  double lambda = 42.0;
  double eps = 1e-12;

  auto G_tau = gf<triqs::mesh::dlr_imtime, scalar_valued>{{beta, Fermion, lambda, eps}};

  double omega = 1.337;
  triqs::clef::placeholder<0> tau_;
  G_tau(tau_) << -nda::clef::exp(-omega * tau_) / (1 + nda::clef::exp(-beta * omega));

  // Density from dlr_imtime not supported (by design)
  //EXPECT_THROW(triqs::gfs::density(G_tau), triqs::runtime_error);
  
  // Density from dlr is efficient (by design)
  // only requires interpolation at \tau=\beta ( n = -G(\beta) )
  auto G_dlr = dlr_coeffs_from_dlr_imtime(G_tau); 
  auto n = triqs::gfs::density(G_dlr);
  EXPECT_COMPLEX_NEAR(n, 1 / (1 + std::exp(beta * omega)), 1.e-9);
}

TEST(Gf, dlr_tau_rev) {

  double beta   = 2.0;
  double lambda = 42.0;
  double eps = 1e-12;

  auto G_tau = gf<triqs::mesh::dlr_imtime, scalar_valued>{{beta, Fermion, lambda, eps}};

  double omega = 1.337;
  triqs::clef::placeholder<0> tau_;
  G_tau(tau_) << nda::clef::exp(-omega * tau_) / (1 + nda::clef::exp(-beta * omega));

  auto G_tau_rev = G_tau; // copy
  
  auto G_dlr = dlr_coeffs_from_dlr_imtime(G_tau);
  G_tau_rev(tau_) << G_dlr(beta - tau_);

  // Interpolation in imaginary time using dlr grid (efficient by design)
  for (auto const &tau : G_tau_rev.mesh()) {
    double val = std::exp(-omega * (beta - tau)) / (1 + std::exp(-beta * omega));
    EXPECT_COMPLEX_NEAR(G_tau_rev[tau].real(), val, eps);
    EXPECT_COMPLEX_NEAR(G_tau_rev[tau], G_dlr(beta - tau), eps);
  }
}

TEST(Gf, dlr_h5) {

  double beta   = 2.0;
  double lambda = 42.0;
  double eps = 1e-12;
  double omega = 1.337;

  triqs::clef::placeholder<0> tau_;

  auto G_tau = gf<triqs::mesh::dlr_imtime, scalar_valued>{{beta, Fermion, lambda, eps}};
  G_tau(tau_) << nda::clef::exp(-omega * tau_) / (1 + nda::clef::exp(-beta * omega));
  auto G_dlr = dlr_coeffs_from_dlr_imtime(G_tau);
  auto G_iw = dlr_imtime_from_dlr_coeffs(G_dlr);

  rw_h5(G_tau, "g_dlr_imtime");
  rw_h5(G_dlr, "g_dlr_coeffs");
  rw_h5(G_iw, "g_dlr_imfreq");
}

TEST(Gf, dlr_dyson) {

  double beta   = 2.0;
  double lambda = 42.0;
  double eps = 1e-12;

  double e1 = 0.0;
  double e2 = 2.0;

  triqs::clef::placeholder<0> tau_;
  triqs::clef::placeholder<1> iw_;

  auto G_tau = gf<triqs::mesh::dlr_imtime, scalar_valued>{{beta, Fermion, lambda, eps}};
  G_tau(tau_) <<
    - 0.5 * nda::clef::exp(-e1 * tau_) / (1 + nda::clef::exp(-beta * e1))
    - 0.5 * nda::clef::exp(-e2 * tau_) / (1 + nda::clef::exp(-beta * e2));

  auto G_dlr = dlr_coeffs_from_dlr_imtime(G_tau);

  auto G_iw = dlr_imfreq_from_dlr_coeffs(G_dlr);

  auto G_iw_ref = G_iw;
  G_iw_ref(iw_) << 0.5/(iw_ - e1) + 0.5/(iw_ - e2);

  for (auto const &iw : G_iw.mesh()) {
    EXPECT_CLOSE(G_iw[iw], G_iw_ref[iw]);
  }

  EXPECT_GF_NEAR(G_iw, G_iw_ref);
}

MAKE_MAIN;
