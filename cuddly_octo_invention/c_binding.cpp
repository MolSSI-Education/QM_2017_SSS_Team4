#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include <vector>
#include <iostream>

namespace py = pybind11;

double dot_product_numpy(py::array_t<double> v1,
			 py::array_t<double> v2)
{
  py::buffer_info v1_info = v1.request();
  py::buffer_info v2_info = v2.request();

  if (v1_info.ndim != 1)
    {
      throw std::runtime_error("v1 is not a vector");
    }
  if (v2_info.ndim != 1)
    {
      throw std::runtime_error("v2 is not a vector");
    }
  if (v1_info.shape[0] != v2_info.shape[0])
    {
      throw std::runtime_error("Vectors are different length");
    }
  double dot = 0.0;

  const double * v1_data = static_cast<double *>(v1_info.ptr);
  const double * v2_data = static_cast<double *>(v2_info.ptr);
n
  for (size_t i = 0; i < v1_info.shape[0]; i++)
    {
      dot += v1_data[i] * v2_data[i];
    }
  return dot;
}


py::array_t<double> dgemm_numpy(double alpha,
				py::array_t<double> A,
				py::array_t<double> B)
{
  py::buffer_info A_info = A.request();
  py::buffer_info B_info = B.request();

  if (A_info.ndim != 2)
    {
      throw std::runtime_error("A is not a matrix");
    }
  if (B_info.ndim != 2)
    {
      throw std::runtime_error("B is not a matrix");
    }
  if (A_info.shape[1] != B_info.shape[0])
    {
      throw std::runtime_error("Matrices are of incompatible sizes");
    }

  size_t C_nrows = A_info.shape[0];
  size_t C_ncols = B_info.shape[1];
  size_t n_k = A_info.shape[1];

  const double * A_data = static_cast<double *>(A_info.ptr);
  const double * B_data = static_cast<double *>(B_info.ptr);
  
  std::vector<double> C_data(C_nrows * C_ncols);

  for (size_t i = 0; i < C_nrows; i++)
    {
      for (size_t j = 0; j < C_ncols; j++)
	{
	  double val = 0.0;
	  for (size_t k = 0; k < n_k; k++)
	    {
	      val += alpha *  A_data[i * n_k + k] * B_data[k * C_ncols + j];
	    }
	  C_data[i * C_ncols + j] = val;
	}
    }
  py::buffer_info Cbuf =
    {
      C_data.data(),
      sizeof(double),
      py::format_descriptor<double>::format(),
      2,
      { C_nrows, C_ncols },
      { C_ncols * sizeof(double), sizeof(double) }
      };
      
      return py::array_t<double>(Cbuf);
}
		   

py::array_t<double> np_einsum(py::array_t<double> Qls_tilde,
			      py::array_t<double> metric,
			      py::array_t<double> D,
			      double bas)
{
  py::buffer_info Qls_tilde_info = Qls_tilde.request();
  py::buffer_info metric_info = metric.request();
  py::buffer_info D_info = D.request();
  	  
  const double * Qtilde = static_cast<double *>(Qls_tilde_info.ptr);
  const double * metric_data = static_cast<double *>(metric_info.ptr);
  
  std::vector<double> Pls(bas, bas, bas);
  std::vector<double> Xp(bas);
  std::vector<double> Pls_t(bas, bas, bas);
  
  std::vector<double> J(bas, bas);
  std::vector<double> eita_data(bas, bas, bas);
  std::vector<double> K_data(bas, bas);

  for ( size_t p=0; p<bas; p++)
{ for( size_t q=0; q<bas; q++)
  { for( size_t l=0; l<bas; l++)
    { for( size_t s=0; s<bas; s++)
     {Pls[p][l][s] += Qtilde[q][l][s]*metric_data[p][q];
     }
    }
  }
}

  for (size_t p=0; p<bas; p++)
{ for(size_t q=0; q<bas; q++)
   {for(size_t r=0; r<bas; r++)
    { Xp[p] += Pls[p][r][q]*D[r][q];
    }
   }
}
  //transpose of pls
for (size_t p=0; p<bas; p++)
{ for(size_t q=0; q<bas; q++)
   {for(size_t r=0; r<bas; r++)
    { Pls_t[l][s][p] = Pls[p][l][s];
    }
   }
}

//writing J
for(size_t l=0; l<bas; l++)
{ for(size_t s=0; s<bas; s++)
  {for(size_t p=0; p<bas; p++)
      {J[l][s] = Pls_t[l][s][p]*Xp[p];
   }
  }
}
  py::buffer_info Jbuf =
    {
      J.data(),
      sizeof(double),
      py::format_descriptor<double>::format(),
      2,
      { bas, bas },
      { bas * sizeof(double), sizeof(double) }
      };
  py::buffer_info Kbuf =
    {
      K_data.data(),
      sizeof(double),
      py::format_descriptor<double>::format(),
      2,
      { buf, buf },
      { buf * sizeof(double), sizeof(double) }
      };
  return Jbuf, Kbuf
}


PYBIND11_PLUGIN(basic_mod)
{
  py::module m("basic_mod", "Kevin's basic module");

  m.def("dot_product_numpy", &dot_product_numpy, "Compute A.B");
  m.def("dgemm_numpy", &dgemm_numpy, "Compute A.B");
  m.def("np_einsum", &np_einsum);
  return m.ptr();
}

