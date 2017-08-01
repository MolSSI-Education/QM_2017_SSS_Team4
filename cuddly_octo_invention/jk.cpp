
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <string>
#include <omp.h>

namespace py = pybind11;

py::array_t<double> jk(py::array_t<double> I,
                               py::array_t<double> D)
{
   py::buffer_info D_info = D.request();
   py::buffer_info I_info = I.request();
   size_t n = D_info.shape[0];

   const double*D_data = static_cast<double*>(D_info.ptr);
   const double*I_data = static_cast<double*>(I_info.ptr);


   std::vector<double> J_data(n*n);
   /*
   JK.resize(2, std::vector<double> (n*n));
	*/
	
   size_t n3 = n*n*n;
   size_t n2 = n*n;
  // Form J
   double start = omp_get_wtime();

#pragma omp parallel for schedule(dynamic) reduction(+:J_data)

   for(size_t p = 0; p < n; p++)
   {
       for(size_t q = p; q < n; q++)
       {
		
           for(size_t r = 0; r < n; r++)
           {
               for(size_t s = r; s < n; s++)
               {

				   if (s == r) 
				    {
						J_data[p*n + q] +=  I_data[p * n3 + q * n2 + r * n + s] * D_data[r * n + s];		
					}
					else
					{
						J_data[p*n + q] += 2 * I_data[p * n3 + q * n2 + r * n + s] * D_data[r * n + s];		
					}
               }
           }
		   
		   J_data[q*n + p] = J_data[p*n + q];
		   
       }
   }

 
   double stop = omp_get_wtime();
   std::cout << "time = " << stop - start << std::endl;
  // Form J
   /*for(size_t p = 0; p < n; p++)
   {
       for(size_t q = 0; q < n; q++)
       {
           for(size_t r = 0; r < n; r++)
           {
               for(size_t s = 0; s < n; s++)
               {
                   J_data[p*n + q] += I_data[p * n3 + q * n2 + r * n + s] * D_data[r * n + s];			   
               }
           }
       }
   }*/   

   py::buffer_info Jbuf =
       {
       J_data.data(),
       sizeof(double),
       py::format_descriptor<double>::format(),
       2,
       {n,n},
       {n * sizeof(double), sizeof(double)}
       };

   return py::array_t<double>(Jbuf);
}



PYBIND11_PLUGIN(jk)
{
   py::module m("jk", "hi");
   m.def("jk", &jk, "this is K");
   return m.ptr();
}
