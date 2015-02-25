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
#include <triqs/utility/is_complex.hpp>
#include "./communicator.hpp"
#include <complex>

namespace triqs {
namespace mpi {

 /// a tag for each operation
 namespace tag {
  struct reduce {};
  struct scatter {};
  struct gather {};
 }

 // The implementation of mpi ops for each type
 // To be specialized later
 template <typename T, typename Enable = void> struct mpi_impl {
  // Broadcast a
  // static void broadcast(T &a, communicator c, int root);

  // Reduces a on site ( all -> all_reduce)
  // static void reduce_in_place(T &a, communicator c, int root, bool all);

  // Return a T or a lazy object
  // static auto invoke(tag::reduce,  T const &a, communicator c, int root, bool all); // all = allreduce
  // static auto invoke(tag::gather,  T const &a, communicator c, int root, bool all); // all = allgather
  // static auto invoke(tag::scatter, T const &a, communicator c, int root, bool); // bool does nothing 

  // _assign (lhs, Tag, c, a, root, all) is the same as lhs = invoke(Tag, c, a, root, all);
  // it implements the operation
  // template <typename Tag> static void _assign(T &lhs, Tag, T const &a, communicator c, int root, bool all);
 };

 // -----------------------------
  
 /// A small lazy tagged class 
 template <typename Tag, typename T> struct mpi_lazy {
  T const &ref;
  communicator c;
  int root;
  bool all;
 };

 // ----------------------------------------
 // ------- top level functions -------
 // ----------------------------------------

 // ----- transmission -------------

 template <typename T> void broadcast(T &x, communicator c = {}, int root = 0) { mpi_impl<T>::broadcast(x, c, root); }

 // to impl: send, recv, ...
 
 // ----- functions that can return lazy object -------

 template <typename T>
 AUTO_DECL reduce(T const &x, communicator c = {}, int root = 0, bool all = false) RETURN(mpi_impl<T>::invoke(tag::reduce(), x, c, root, all));
 template <typename T>
 AUTO_DECL all_reduce(T const &x, communicator c = {}, int root = 0) RETURN(mpi_impl<T>::invoke(tag::reduce(), x, c, root, true));
 template <typename T>
 AUTO_DECL scatter(T const &x, communicator c = {}, int root = 0) RETURN(mpi_impl<T>::invoke(tag::scatter(), x, c, root, true));
 template <typename T>
 AUTO_DECL gather(T const &x, communicator c = {}, int root = 0) RETURN(mpi_impl<T>::invoke(tag::gather(), x, c, root, false));
 template <typename T>
 AUTO_DECL all_gather(T const &x, communicator c = {}, int root = 0) RETURN(mpi_impl<T>::invoke(tag::gather(), x, c, root, true));

 // ----- short cuts -------
/*
 template <typename T> void reduce_in_place(T &x, communicator c = {}, int root = 0, bool all = false) {
  x = reduce(x, c, root, all); 
 }
 template <typename T> void all_reduce_in_place(T &x, communicator c = {}, int root = 0) { reduce_in_place(x, c, root, true); }
*/

 // impl. detail : internal use only, to deduce T
 template <typename T, typename Tag> void _assign(T &lhs, Tag, T const &rhs, communicator c, int root, bool all) {
   mpi_impl<T>::_assign(lhs, Tag(), rhs, c, root, all);
 }

 /** ------------------------------------------------------------
   *  transformation type -> mpi types
   *  ----------------------------------------------------------  **/

 template <class T> MPI_Datatype mpi_datatype();
#define D(T, MPI_TY)                                                                                                             \
 template <> MPI_Datatype mpi_datatype<T> { return MPI_TY; }                                                                     \
 }                                                                                                                               \
 ;
 D(int, MPI_INT) D(long, MPI_LONG) D(double, MPI_DOUBLE) D(float, MPI_FLOAT) D(std::complex<double>, MPI_DOUBLE_COMPLEX);
 D(unsigned long, MPI_UNSIGNED_LONG); D(unsigned int, MPI_UNSIGNED); D(unsigned long long, MPI_UNSIGNED_LONG_LONG);
#undef D

 /** ------------------------------------------------------------
   *  basic types
   *  ----------------------------------------------------------  **/

 template <typename T> struct mpi_impl_basic {

  static void broadcast(T &a, communicator c, int root) { MPI_Bcast(&a, 1, mpi_datatype<T>(), root, c.get()); }

  static T invoke(tag::reduce, communicator c, T a, int root, bool all) {
   T b;
   auto d = mpi_datatype<T>();
   if (!all)
    MPI_Reduce(&a, &b, 1, d, MPI_SUM, root, c.get());
   else
    MPI_Allreduce(&a, &b, 1, d, MPI_SUM, c.get());
   return b;
  }

  template <typename Tag> static void _assign(T &lhs, Tag, T a, communicator c, int root) { lhs = invoke(Tag(), c, a, root); }
 };

 // mpl_impl_basic is the mpi_impl<T> is T is a number (including complex)
 template <typename T>
 struct mpi_impl<T, std14::enable_if_t<std::is_arithmetic<T>::value || triqs::is_complex<T>::value>> : mpi_impl_basic<T> {};

 //------------ Some helper function

 // Given a range [first, last], slice it regularly for a node of rank 'rank' among n_nodes.
 // If the range is not dividable in n_nodes equal parts,
 // the first nodes have one more elements than the last ones.
 inline std::pair<long, long> slice_range(long first, long last, int n_nodes, int rank) {
  long chunk = (last - first + 1) / n_nodes;
  long n_large_nodes = (last - first + 1) - n_nodes * chunk;
  if (rank <= n_large_nodes - 1) // first, larger nodes, use chunk + 1
   return {first + rank * (chunk + 1), first + (rank + 1) * (chunk + 1) - 1};
  else // others nodes : shift the first by 1*n_large_nodes, used chunk
   return {first + n_large_nodes + rank * chunk, first + n_large_nodes + (rank + 1) * chunk - 1};
 }

 inline long slice_length(long imax, int n_nodes, int rank) {
  auto r = slice_range(0, imax, n_nodes, rank);
  return r.second - r.first + 1;
 }

}
}
