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
#include "./base.hpp"
#include <triqs/utility/tuple_tools.hpp>

#define TRIQS_MPI_AS_TUPLE using triqs_mpi_as_tuple = void;
#define TRIQS_MPI_AS_TUPLE_LAZY using triqs_mpi_as_tuple_lazy = void;
namespace triqs {
namespace mpi {

 template <typename T> auto get_mpi_tuple(T&x) RETURN(x.get_mpi_tuple());

 struct __reduce_lambda {
  communicator c;
  int root, all;
  template <typename U> void operator()(U &x) const { mpi_impl<U>::reduce_in_place(c, x, root, all); }
 };

 template <typename F> struct lambda_to_archive {
  F f;
  template <typename T> lambda_to_archive &operator&(T &x) const { f(x); return *this; }
 };
 template <typename F> lambda_to_archive<F> make_lambda_to_archive(F &&f) { return std::forward<F>(f); }

 struct __bcast_lambda {
  communicator c;
  int root;
  template <typename U> void operator()(U &x) const { triqs::mpi::broadcast(x, c, root); }
 };

 template <typename Tag> struct __2_lambda {
  communicator c;
  int root;
  template <typename U1, typename U2> void operator()(U1 &u1, U2 &u2) const { triqs::mpi::_assign(u1, Tag(), c, u2, root); }
 };
/*
 template <> struct __2_lambda<tag::reduce> {
  communicator c;
  int root;
  template <typename U1, typename U2> void operator()(U1 &u1, U2 &u2) const {
   triqs::mpi::_assign(u1, tag::reduce(), c, u2, root);
   u1 = u2;
  }
 };

 template <> struct __2_lambda<tag::all_reduce> {
  communicator c;
  int root;
  template <typename U1, typename U2> void operator()(U1 &u1, U2 &u2) const {
   triqs::mpi::_assign(u1, tag::all_reduce(), c, u2, root);
  }
  template <typename U1, typename U2> void operator()(__no_reduction<U1> &u1, __no_reduction<U2> &u2) const {
   u1 = u2;
   triqs::mpi::broadcast(__strip(u1), c, root);
  }
 };
*/
 // to allow to access the data...
 struct access { 
  template<typename T>
  static void broadcast(communicator c, T &a, int root) {
   auto f = make_lambda_to_archive(__bcast_lambda{c,root});
   a.serialize(f,0);
  }
 };

 /** ------------------------------------------------------------
  *  Type which are recursively treated by reducing them to a tuple
  *  of smaller objects.
  *  ----------------------------------------------------------  **/
 template <typename T> struct mpi_impl_tuple_lazy : access {

  static void reduce_in_place(communicator c, T &a, int root, bool all) {
   tuple::for_each(get_mpi_tuple_reduction(a), __reduce_lambda{c, root, all});
  }

  template <typename Tag> static mpi_lazy<Tag, T> invoke(Tag, communicator c, T const &a, int root) {
   return {a, root, c};
  }

  template <typename Tag> static void complete_operation(T &lhs, mpi_lazy<Tag, T> laz) {
   assign(lhs, Tag(), laz.c, laz.ref, laz.root);
  }

  template <typename Tag> static void _assign(T &target, Tag, communicator c, T const &x, int root) {
   triqs::tuple::for_each_zip(__2_lambda<Tag>{c, root}, get_mpi_tuple_reduction(target), get_mpi_tuple_reduction(x));
  }
 };

 // -----------------------------------------------------------

 // no lazy, overrule the invoke.
 template <typename T> struct mpi_impl_tuple : mpi_impl_tuple_lazy<T>  {

  template <typename Tag> static T invoke(Tag, communicator c, T const &a, int root) {
   T b = a;
   mpi_impl_tuple_lazy<T>::_assign(b, Tag(), c, a, root);
   return b;
  }
 };

 // If type T has a mpi_implementation nested struct, then it is mpi_impl<T>.
 template <typename T> struct mpi_impl<T, typename T::triqs_mpi_as_tuple_lazy> : mpi_impl_tuple_lazy<T> {};
 template <typename T> struct mpi_impl<T, typename T::triqs_mpi_as_tuple> : mpi_impl_tuple<T> {};
}
} // namespace

