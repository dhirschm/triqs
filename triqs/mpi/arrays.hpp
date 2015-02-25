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
  int root;
  communicator c;

  using domain_type = typename A::domain_type;

  /// compute the array domain of the target array
  domain_type domain() const {
   auto dims = ref.shape();
   long slow_size = first_dim(ref);

   // tag::reduce and all_reduce : do nothing
 
   if (std::is_same<Tag, tag::scatter>::value) {
    mpi::broadcast(slow_size, c, root);
    dims[0] = mpi::slice_length(slow_size - 1, c.size(), c.rank());
   }
   
   if (std::is_same<Tag, tag::gather>::value) {
    auto s = mpi::reduce(slow_size, c, root); 
    dims[0] = (c.rank()==root ? s : 1); // valid only on root
   }
   
   if (std::is_same<Tag, tag::allgather>::value) {
    dims[0] = mpi::all_reduce(slow_size, c, root); // in this case, it is valid on all nodes
   }
   
   return domain_type{dims};
  }

 };

 //--------------------------------------------------------------------------------------------------------

 // When value_type is a basic type, we can directly call the C API
 template <typename A> class mpi_impl_triqs_arrays {

  static MPI_Datatype D() { return mpi_datatype<typename A::value_type>::invoke(); }

  static void check_is_contiguous(A const &a) {
   if (!has_contiguous_data(a)) TRIQS_RUNTIME_ERROR << "Non contiguous view in mpi_reduce_in_place";
  }

  public:

  //---------
  static void reduce_in_place(communicator c, A &a, int root, bool all) {
   check_is_contiguous(a);
   // assume arrays have the same size on all nodes...
   if (!all)
    MPI_Reduce((c.rank() == root ? MPI_IN_PLACE : a.data_start()), a.data_start(), a.domain().number_of_elements(), D(), MPI_SUM, root,
               c.get());
   else
    MPI_Allreduce(MPI_IN_PLACE, a.data_start(), a.domain().number_of_elements(), D(), MPI_SUM, c.get());
  }

  //---------
  static void broadcast(communicator c, A &a, int root) {
   check_is_contiguous(a);
   auto sh = a.shape();
   MPI_Bcast(&sh[0], sh.size(), mpi_datatype<typename decltype(sh)::value_type>::invoke(), root, c.get());
   if (c.rank() != root) a.resize(sh);
   MPI_Bcast(a.data_start(), a.domain().number_of_elements(), D(), root, c.get());
  }

  //---------
  template <typename Tag> static mpi_lazy_array<Tag, A> invoke(Tag, communicator c, A const &a, int root) {
   check_is_contiguous(a);
   return {a, root, c};
  }

  template <typename Tag> static void _assign(A & lhs, Tag, communicator c, A const &a, int root) {
   lhs = invoke(Tag(), c, a, root);
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
   static MPI_Datatype D() { return mpi::mpi_datatype<typename A::value_type>::invoke(); }

   //---------------------------------
   void _invoke(triqs::mpi::tag::reduce) {
    if (laz.c.rank() == laz.root) lhs.resize(laz.domain());
    MPI_Reduce((void *)laz.ref.data_start(), (void *)lhs.data_start(), laz.ref.domain().number_of_elements(), D(), MPI_SUM, laz.root, laz.c.get());
   }

   //---------------------------------
   void _invoke(triqs::mpi::tag::all_reduce) {
    // ADD debug check under macro that all nodes have same size
    lhs.resize(laz.domain());
    MPI_Allreduce((void *)laz.ref.data_start(), (void *)lhs.data_start(), laz.ref.domain().number_of_elements(), D(), MPI_SUM, laz.c.get());
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
    auto d = laz.domain();
    if (laz.c.rank() == laz.root) lhs.resize(d);

    auto c = laz.c;
    auto recvcounts = std::vector<int>(c.size());
    auto displs = std::vector<int>(c.size() + 1, 0);
    int sendcount = laz.ref.domain().number_of_elements();

    auto mpi_ty = mpi::mpi_datatype<int>::invoke();
    MPI_Gather(&sendcount, 1, mpi_ty, &recvcounts[0], 1, mpi_ty, laz.root, c.get());
    for (int r = 0; r < c.size(); ++r) displs[r + 1] = recvcounts[r] + displs[r];

    MPI_Gatherv((void *)laz.ref.data_start(), sendcount, D(), (void *)lhs.data_start(), &recvcounts[0], &displs[0], D(), laz.root,
                c.get());
   }

   //---------------------------------
   void _invoke(triqs::mpi::tag::allgather) {
    lhs.resize(laz.domain());

    // almost the same preparation as gather, except that the recvcounts are ALL gathered...
    auto c = laz.c;
    auto recvcounts = std::vector<int>(c.size());
    auto displs = std::vector<int>(c.size() + 1, 0);
    int sendcount = laz.ref.domain().number_of_elements();

    auto mpi_ty = mpi::mpi_datatype<int>::invoke();
    MPI_Allgather(&sendcount, 1, mpi_ty, &recvcounts[0], 1, mpi_ty, c.get());
    for (int r = 0; r < c.size(); ++r) displs[r + 1] = recvcounts[r] + displs[r];

    MPI_Allgatherv((void *)laz.ref.data_start(), sendcount, D(), (void *)lhs.data_start(), &recvcounts[0], &displs[0], D(),
                   c.get());
   }
  };
 }
} //namespace arrays
} // namespace triqs
