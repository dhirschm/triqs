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

 template <typename T> struct __no_reduction {
  T &x;
 };
 template <typename T> __no_reduction<T> no_reduction(T &x) {
  return {x};
 }

 template <typename T> T &__strip(__no_reduction<T> x) { return x.x; }
 template <typename T> T &__strip(T &x) { return x; }

 struct __reduce_lambda {
  communicator c;
  int root, all;
  template <typename U> void operator()(U &x) const { mpi_impl<U>::reduce_in_place(c, x, root, all); }
  template <typename U> void operator()(__no_reduction<U>) const {}
 };

 struct __bcast_lambda {
  communicator c;
  int root;
  template <typename U> void operator()(U &x) const { triqs::mpi::broadcast(__strip(x), c, root); }
 };

 template <typename Tag> struct __2_lambda {
  communicator c;
  int root;
  template <typename U1, typename U2> void operator()(U1 &u1, U2 &u2) const {
   triqs::mpi::_invoke2(__strip(u1), Tag(), c, __strip(u2), root);
  }
 };

 template <> struct __2_lambda<tag::reduce> {
  communicator c;
  int root;
  template <typename U1, typename U2> void operator()(U1 &u1, U2 &u2) const {
   triqs::mpi::_invoke2(__strip(u1), tag::reduce(), c, __strip(u2), root);
  }
  template <typename U1, typename U2> void operator()(__no_reduction<U1> &u1, __no_reduction<U2> &u2) const {
   //if (c.rank() == root)  // no, leads to a bug with tail ...
   __strip(u1) = __strip(u2);
  }
 };

 template <> struct __2_lambda<tag::all_reduce> {
  communicator c;
  int root;
  template <typename U1, typename U2> void operator()(U1 &u1, U2 &u2) const {
   triqs::mpi::_invoke2(u1, tag::all_reduce(), c, u2, root);
  }
  template <typename U1, typename U2> void operator()(__no_reduction<U1> &u1, __no_reduction<U2> &u2) const {
   __strip(u1) = __strip(u2);
   triqs::mpi::broadcast(__strip(u1), c, root);
  }
 };

 /** ------------------------------------------------------------
  *  Type which are recursively treated by reducing them to a tuple
  *  of smaller objects.
  *  ----------------------------------------------------------  **/
 template <typename T> struct mpi_impl_tuple_lazy {

  static void reduce_in_place(communicator c, T &a, int root, bool all) {
   tuple::for_each(get_mpi_tuple(a), __reduce_lambda{c, root, all});
  }

  static void broadcast(communicator c, T &a, int root) {
   tuple::for_each(get_mpi_tuple(a), __bcast_lambda{c, root});
  }

  template <typename Tag> static mpi_lazy<Tag, T> invoke(Tag, communicator c, T const &a, int root) {
   return {a, root, c};
  }

  template <typename Tag> static void complete_operation(T &lhs, mpi_lazy<Tag, T> laz) {
   invoke2(lhs, Tag(), laz.c, laz.ref, laz.root);
  }

  template <typename Tag> static void invoke2(T &target, Tag, communicator c, T const &x, int root) {
   triqs::tuple::for_each_zip(__2_lambda<Tag>{c, root}, get_mpi_tuple(target), get_mpi_tuple(x));
  }
 };

 // -----------------------------------------------------------

 // no lazy, overrule the invoke.
 template <typename T> struct mpi_impl_tuple : mpi_impl_tuple_lazy<T>  {

  template <typename Tag> static T invoke(Tag, communicator c, T const &a, int root) {
   T b = a;
   mpi_impl_tuple_lazy<T>::invoke2(b, Tag(), c, a, root);
   return b;
  }
 };

 // If type T has a mpi_implementation nested struct, then it is mpi_impl<T>.
 template <typename T> struct mpi_impl<T, typename T::triqs_mpi_as_tuple_lazy> : mpi_impl_tuple_lazy<T> {};
 template <typename T> struct mpi_impl<T, typename T::triqs_mpi_as_tuple> : mpi_impl_tuple<T> {};
}
} // namespace

