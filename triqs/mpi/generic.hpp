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

#define TRIQS_MPI_IMPLEMENTED_AS_TUPLEVIEW using triqs_mpi_as_tuple = void;
#define TRIQS_MPI_IMPLEMENTED_AS_TUPLEVIEW_NO_LAZY using triqs_mpi_as_tuple_no_lazy = void;
namespace triqs {
namespace mpi {

 /** ------------------------------------------------------------
  *  Type which are recursively treated by reducing them to a tuple
  *  of smaller objects.
  *  ----------------------------------------------------------  **/
 template <typename T, bool with_lazy> struct mpi_impl_tuple {

  mpi_impl_tuple() = default;

  /// invoke
  template <typename Tag> static mpi_lazy<Tag, T> invoke_impl(std::true_type, Tag, communicator c, T const &a, int root) {
   return {a, root, c};
  }
  
  template <typename Tag> static T &invoke_impl(std::false_type, Tag, communicator c, T const &a, int root) {
   return complete_operation(a, {a, root, c});
  }

  template <typename Tag> static mpi_lazy<Tag, T> invoke(Tag, communicator c, T const &a, int root) {
   return invoke_impl(std::integral_constant<bool, with_lazy>(), Tag(), c, a, root);
  }

  static void reduce_in_place(communicator c, T &a, int root) {
   tuple::for_each(view_as_tuple(a), [c, root](auto &x) { triqs::mpi::reduce_in_place(x, c, root); });
  }

  static void broadcast(communicator c, T &a, int root) {
   tuple::for_each(view_as_tuple(a), [c, root](auto &x) { triqs::mpi::broadcast(x, c, root); });
  }

  template <typename Tag> static T &complete_operation(T &target, mpi_lazy<Tag, T> laz) {
   auto l = [laz](auto &t, auto &s) { t = triqs::mpi::mpi_impl<std::decay_t<decltype(s)>>::invoke(Tag(), laz.c, s, laz.root); };
   triqs::tuple::for_each_zip(l, view_as_tuple(target), view_as_tuple(laz.ref));
   return target;
  }
 };

 // If type T has a mpi_implementation nested struct, then it is mpi_impl<T>.
 template <typename T> struct mpi_impl<T, typename T::triqs_mpi_as_tuple> : mpi_impl_tuple<T, true> {};
 template <typename T> struct mpi_impl<T, typename T::triqs_mpi_as_tuple_no_lazy> : mpi_impl_tuple<T, false> {};
}
} // namespace

