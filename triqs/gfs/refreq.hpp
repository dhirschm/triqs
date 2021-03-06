/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2012-2013 by O. Parcollet
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
#include "./tools.hpp"
#include "./gf.hpp"
#include "./local/tail.hpp"
#include "./meshes/segment.hpp"
#include "./evaluators.hpp"

namespace triqs {
namespace gfs {

 struct refreq {};

 template <> struct gf_mesh<refreq> : segment_mesh {
  template <typename... T> gf_mesh(T &&... x) : segment_mesh(std::forward<T>(x)...) {}
  //using segment_mesh::segment_mesh;
 };

 // singularity
 template <> struct gf_default_singularity<refreq, matrix_valued> {
  using type = tail;
 };
 template <> struct gf_default_singularity<refreq, scalar_valued> {
  using type = tail;
 };

  namespace gfs_implementation {

  // h5 name
  template <typename Singularity> struct h5_name<refreq, matrix_valued, Singularity> {
   static std::string invoke() { return "ReFreq"; }
  };

  /// ---------------------------  evaluator ---------------------------------

  template <> struct evaluator_of_clef_expression<refreq> : evaluator_grid_linear_interpolation {};

  template <typename Singularity, typename Target> struct evaluator<refreq, Target, Singularity> : evaluator_one_var<refreq> {
   template <typename G> evaluator(G *) {};
  };

  /// ---------------------------  data access  ---------------------------------
  template <> struct data_proxy<refreq, matrix_valued> : data_proxy_array<std::complex<double>, 3> {};
  template <> struct data_proxy<refreq, scalar_valued> : data_proxy_array<std::complex<double>, 1> {};
 }

 // FOR LEGACY PYTHON CODE ONLY
 // THIS MUST be kept for python operations
 // specific operations (for legacy python code).
 // +=, -= with a matrix
 inline void operator+=(gf_view<refreq> g, arrays::matrix<std::complex<double>> const &m) {
  for (int u = 0; u < int(first_dim(g.data())); ++u) g.data()(u, arrays::ellipsis()) += m;
  g.singularity()(0) += m;
 }

 inline void operator-=(gf_view<refreq> g, arrays::matrix<std::complex<double>> const &m) {
  for (int u = 0; u < int(first_dim(g.data())); ++u) g.data()(u, arrays::ellipsis()) -= m;
  g.singularity()(0) -= m;
 }

 inline void operator+=(gf_view<refreq> g, std::complex<double> a) {
  operator+=(g, arrays::make_unit_matrix(get_target_shape(g)[0], a));
 }
 inline void operator-=(gf_view<refreq> g, std::complex<double> a) {
  operator-=(g, arrays::make_unit_matrix(get_target_shape(g)[0], a));
 }


 inline gf<refreq> operator+(gf<refreq> g, arrays::matrix<std::complex<double>> const &m) {
  g() += m;
  return g;
 }

 inline gf<refreq> operator+(gf<refreq> g, std::complex<double> const &m) {
  g() += m; // () is critical of infinite loop -> segfault
  return g;
 }

 inline gf<refreq> operator+(std::complex<double> const &m, gf<refreq> g) { return g + m; }
 inline gf<refreq> operator+(arrays::matrix<std::complex<double>> const &m, gf<refreq> g) { return g + m; }

 inline gf<refreq> operator-(gf<refreq> g, arrays::matrix<std::complex<double>> const &m) {
  g() -= m;
  return g;
 }

 inline gf<refreq> operator-(gf<refreq> g, std::complex<double> const &m) {
  g() -= m;
  return g;
 }

 inline gf<refreq> operator-(std::complex<double> const &m, gf<refreq> g) { 
  g *= -1;
  g+=m;
  return g;
  }

 inline gf<refreq> operator-(arrays::matrix<std::complex<double>> const &m, gf<refreq> g) { 
  g *= -1;
  g+=m;
  return g;
 }
}
}

