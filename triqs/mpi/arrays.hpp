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

namespace triqs {
namespace mpi {

 //--------------------------------------------------------------------------------------------------------
 // The lazy ref made by scatter and co. 
 // Differs from the generic one in that it can make a domain of the (target) array
 template <typename Tag, typename A> struct mpi_lazy_array {
  A const &ref;
  communicator c;
  int root;
  bool all;

  using domain_type = typename A::domain_type;

  /// compute the array domain of the target array
  domain_type domain() const {
   auto dims = ref.shape();
   long slow_size = first_dim(ref);
 
   if (std::is_same<Tag, tag::scatter>::value) {
    mpi::broadcast(slow_size, c, root);
    dims[0] = mpi::slice_length(slow_size - 1, c.size(), c.rank());
   }

   if (std::is_same<Tag, tag::gather>::value) {
    if (!all)
     dims[0] = (c.rank() == root ? mpi::reduce(slow_size, c, root) : 1); // valid only on root
    else
     dims[0] = mpi::all_reduce(slow_size, c, root); // in this case, it is valid on all nodes
   }
   // tag::reduce :do nothing

   return domain_type{dims};
  }
 };

 //--------------------------------------------------------------------------------------------------------

 // When value_type is a basic type, we can directly call the C API
 template <typename A> class mpi_impl_triqs_arrays {

  static MPI_Datatype D() { return mpi_datatype<typename A::value_type>(); }

  static void check_is_contiguous(A const &a) {
   if (!has_contiguous_data(a)) TRIQS_RUNTIME_ERROR << "Non contiguous view in mpi_reduce_in_place";
  }

  public:
  //---------
  static void broadcast(A &a, communicator c, int root) {
   check_is_contiguous(a);
   auto sh = a.shape();
   MPI_Bcast(&sh[0], sh.size(), mpi_datatype<typename decltype(sh)::value_type>(), root, c.get());
   if (c.rank() != root) a.resize(sh);
   MPI_Bcast(a.data_start(), a.domain().number_of_elements(), D(), root, c.get());
  }

  //---------
  template <typename Tag> static mpi_lazy_array<Tag, A> invoke(Tag, A const &a, communicator c, int root, bool all) {
   check_is_contiguous(a);
   return {a, c, root, all};
  }

  template <typename Tag> static void _assign(A &lhs, Tag, A const &a, communicator c, int root) {
   lhs = invoke(Tag(), a, c, root);
  }
 };

 template <typename A>
 struct mpi_impl<A, std14::enable_if_t<triqs::arrays::is_amv_value_or_view_class<A>::value>> : mpi_impl_triqs_arrays<A> {};

} // mpi namespace 

//------------------------------- Delegation of the assign operator of the array class -------------

namespace arrays {

 // mpi_lazy_array model ImmutableCuboidArray
 template <typename Tag, typename A> struct ImmutableCuboidArray<mpi::mpi_lazy_array<Tag, A>> : ImmutableCuboidArray<A> {};

 namespace assignment {

  template <typename LHS, typename Tag, typename A> struct is_special<LHS, mpi::mpi_lazy_array<Tag, A>> : std::true_type {};

  // assignment delegation
  template <typename LHS, typename A, typename Tag> struct impl<LHS, mpi::mpi_lazy_array<Tag, A>, 'E', void> {

   using laz_t = mpi::mpi_lazy_array<Tag, A>;
   LHS &lhs;
   laz_t laz;

   impl(LHS &lhs_, laz_t laz_) : lhs(lhs_), laz(laz_) {}

   void invoke() { _invoke(Tag()); }

   private:
   static MPI_Datatype D() { return mpi::mpi_datatype<typename A::value_type>(); }

   //---------------------------------
   void _invoke(triqs::mpi::tag::reduce) {

    auto rhs_n_elem = laz.ref.domain().number_of_elements();
    void *lhs_p = lhs.data_start();
    void *rhs_p = laz.ref.data_start();
    auto c = laz.c;
    auto root = laz.root;

    bool in_place = (lhs_p == rhs_p); // to be refined. Overlapping condition
    // some checks.
    if (in_place) {
     if (rhs_n_elem != lhs.domain().number_of_elements())
      TRIQS_RUNTIME_ERROR << "mpi reduce of array : same pointer to data start, but differnet number of elements !";
    } else { // check no overlap
     if (std::abs(lhs.data_start() - laz.ref.data_start()) <= rhs_n_elem)
      TRIQS_RUNTIME_ERROR << "mpi reduce of array : overlapping arrays !";
    }

    if (!laz.all) {
     if (in_place)
      MPI_Reduce((c.rank() == root ? MPI_IN_PLACE : rhs_p), rhs_p, rhs_n_elem, D(), MPI_SUM, root, c.get());
     else {
      if (c.rank() == root) lhs.resize(laz.domain());
      MPI_Reduce(rhs_p, lhs_p, rhs_n_elem, D(), MPI_SUM, root, c.get());
     }
    } else { // all reduce
     if (in_place)
      MPI_Allreduce(MPI_IN_PLACE, rhs_p, rhs_n_elem, D(), MPI_SUM, c.get());
     else {
      lhs.resize(laz.domain());
      MPI_Allreduce(rhs_p, lhs_p, rhs_n_elem, D(), MPI_SUM, c.get());
     }
    }
   }

   //---------------------------------
   void _invoke(triqs::mpi::tag::scatter) {
    lhs.resize(laz.domain());

    auto c = laz.c;
    auto slow_size = first_dim(laz.ref);
    auto slow_stride = laz.ref.indexmap().strides()[0];
    auto sendcounts = std::vector<int>(c.size());
    auto displs = std::vector<int>(c.size() + 1, 0);
    int recvcount = mpi::slice_length(slow_size - 1, c.size(), c.rank()) * slow_stride;

    for (int r = 0; r < c.size(); ++r) {
     sendcounts[r] = mpi::slice_length(slow_size - 1, c.size(), r) * slow_stride;
     displs[r + 1] = sendcounts[r] + displs[r];
    }

    MPI_Scatterv((void *)laz.ref.data_start(), &sendcounts[0], &displs[0], D(), (void *)lhs.data_start(), recvcount, D(),
                 laz.root, c.get());
   }

   //---------------------------------
   void _invoke(triqs::mpi::tag::gather) {
    auto c = laz.c.get();
    auto recvcounts = std::vector<int>(c.size());
    auto displs = std::vector<int>(c.size() + 1, 0);
    int sendcount = laz.ref.domain().number_of_elements();
    void *lhs_p = lhs.data_start();
    void *rhs_p = laz.ref.data_start();

    if (laz.all || (laz.c.rank() == laz.root)) lhs.resize(laz.domain());

    auto mpi_ty = mpi::mpi_datatype<int>();
    if (!laz.all)
     MPI_Gather(&sendcount, 1, mpi_ty, &recvcounts[0], 1, mpi_ty, laz.root, c);
    else
     MPI_Allgather(&sendcount, 1, mpi_ty, &recvcounts[0], 1, mpi_ty, c);

    for (int r = 0; r < c.size(); ++r) displs[r + 1] = recvcounts[r] + displs[r];

    if (!laz.all)
     MPI_Gatherv(rhs_p, sendcount, D(), lhs_p, &recvcounts[0], &displs[0], D(), laz.root, c);
    else
     MPI_Allgatherv(rhs_p, sendcount, D(), lhs_p, &recvcounts[0], &displs[0], D(), c);
   }

  };
 }
} //namespace arrays
} // namespace triqs
