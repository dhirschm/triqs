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
#include "./arrays.hpp"
#include "./vector.hpp"

namespace triqs {
namespace mpi {

 //--------------------------------------------------------------------------------------------------------

 // When value_type is a basic type, we can directly call the C API
 template <typename G> struct mpi_impl_triqs_gfs {

  //---------
  static void broadcast(G &g, communicator c, int root) {
   // Shall we bcast mesh ?
   triqs::mpi::broadcast(g.data(), c, root);
   triqs::mpi::broadcast(g.singularity(), c, root);
  }

  //---------
  template <typename Tag> static mpi_lazy<Tag, G> invoke(Tag, G const &g, communicator c, int root) {
   return {g, root, c};
  }

  //---------
  // REMOVE : change gf
  template <typename Tag> static void complete_operation(G &lhs, mpi_lazy<Tag, G> laz) {
   _assign(lhs, Tag(), laz.c, laz.ref, laz.root);
  }

  static void complete_operation(...) { // impl pour les  tags
  }

  static void reduce(...) ;
  
  //---- reduce ----
  static void _assign(G &lhs, tag::reduce, G const &g, communicator c, int root, bool all) {
   lhs._mesh = g._mesh;
   mpi::_assign(lhs._data, tag::reduce(), c, g.data(), root, all);
   mpi::_assign(lhs._singularity, tag::reduce(), c, g.singularity(), root, all);
  }

  //---- scatter ----
  static void _assign(G &lhs, tag::scatter, G const &g, communicator c, int root, bool) {
   lhs._mesh = mpi_scatter(g.mesh(), c, root);
   mpi::_assign(lhs._data, tag::scatter(), c, g.data(), root, true);
   if (c.rank() == root) lhs._singularity = g.singularity();
   mpi::broadcast(lhs._singularity, c, root);
  }

  //---- gather ----
  static void _assign(G &lhs, tag::gather, G const &g, communicator c, int root, bool all) {
   lhs._mesh = mpi_gather(g.mesh(), c, root);
   mpi::_assign(lhs._data, tag::gather(), c, g.data(), root, all);
   if (all || (c.rank() == root)) lhs._singularity = g.singularity();
  }
 };

 // ---------------------------------------------------------------------------------------
 //  Do nothing for nothing...
 // ---------------------------------------------------------------------------------------
 template <> struct mpi_impl<gfs::nothing> {
  template <typename Tag> static void _assign(gfs::nothing &lhs, Tag, communicator c, gfs::nothing const &a, int root) {}
  template <typename Tag> static gfs::nothing invoke(Tag, communicator c, gfs::nothing const &a, int root) { return gfs::nothing(); }
  static void broadcast(communicator c, gfs::nothing &a, int root) {}
 };

} // mpi namespace
} // namespace triqs
