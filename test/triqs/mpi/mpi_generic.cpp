/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2013 by O. Parcollet
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
#include <iostream>
#include <type_traits>
#include <triqs/arrays.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <triqs/mpi/generic.hpp>

using namespace triqs;
using namespace triqs::arrays;
using namespace triqs::mpi;

struct my_object {

 array<double, 1> a, b;

 TRIQS_MPI_AS_TUPLE;

 my_object() = default;

 my_object(int s) : a(s), b(s) {
  clef::placeholder<0> i_;
  a(i_) << i_;
  b(i_) << -i_;
 }

 // construction from the lazy is delegated to =
 //template <typename Tag> my_object(mpi_lazy<Tag, my_object> x) : my_object() { operator=(x); }

 // assigment is almost done already...
 //template <typename Tag> my_object &operator=(mpi_lazy<Tag, my_object> x) {
 // return mpi_impl<my_object>::assign(*this, x);
// }

 template<typename Archive> 
  void serialize(Archive & A, const unsigned version) { 
   A & a & b;
  }

 // si la structure existe, alors c'est le default, avec void_t
 struct mpi_impl {
  TRIQS_MPI_TRANSMIT_VIA_SERIALIZE; // TOUJOUR SVRAI
  TRIQS_MPI_LAZY; // implemente complete_op, etc...

// 1 struct qui implemente reduce, gather, scatter 
// et un static bool lazy  
// Then rebuild mpi_impl
  static void reduce(G &lhs, G const &rhs, communicator c, int root, bool all) {
   lhs.a = mpi::reduce(rhs.a, c, root, all);
   lhs.b = rhs.b; // master only ?
  }

 };

};

// non intrusive 
auto get_mpi_tuple_reduction(my_object const &x) RETURN(std::make_tuple(std::ref(x.a)));
auto get_mpi_tuple_reduction(my_object &x)       RETURN(std::make_tuple(std::ref(x.a)));

// --------------------------------------

int main(int argc, char *argv[]) {

 mpi::environment env(argc, argv);
 mpi::communicator world;
 
 std::ofstream out("t2_node" + std::to_string(world.rank()));

 auto ob = my_object(10);
 mpi::broadcast(ob);
 
 out << "  a = " << ob.a << std::endl;
 out << "  b = " << ob.b << std::endl;
 
 auto ob2 = ob;

 // ok scatter all components
 ob2 = mpi::scatter(ob);

 out << " scattered  a = " << ob2.a << std::endl;
 out << " scattered  b = " << ob2.b << std::endl;

 ob2.a *= world.rank()+1; // change it a bit

 // now regroup...
 ob = mpi::gather(ob2);
 out << " gather a = " << ob.a << std::endl;
 out << " gather b = " << ob.b << std::endl;

 // allgather
 ob = mpi::allgather(ob2);
 out << " allgather a = " << ob.a << std::endl;
 out << " allgather b = " << ob.b << std::endl;

 // reduce 
 auto ob3 = ob; 
 mpi::reduce_in_place(ob3);
 out << " reduce a = " << ob3.a << std::endl;
 out << " reduce b = " << ob3.b << std::endl;


 out << "----------------------------"<< std::endl;
}

