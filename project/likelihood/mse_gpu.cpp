#include "spectrum.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void sfs_mse_expectation(Spectrum::MSE & mse, const double * gamma, const double * h, const double F, const double * proportion, const int num_categories, const double theta, const double num_sites);

py::array_t<double> normalize_SFS(const float * SFS_in, const int SFS_in_size, py::array_t<const double> & alpha, Spectrum::MSE & mse){	
	py::buffer_info buf = alpha.request();
    
    if (buf.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");
    
    int SFS_size = mse.SFS_size;
    int SFS_unfolded =  mse.sample_size;
    bool zero_class = mse.zero_class;
        
    if (buf.size != SFS_size - 2)
    	throw std::runtime_error("Alpha size must be 2 less than SFS_size");
    
    const double * a_ptr = (const double *)buf.ptr;
    	
    auto SFS_out = py::array_t<double>(SFS_size);
	py::buffer_info buf2 = SFS_out.request();
	auto out_ptr = (double *)buf2.ptr;
    
	bool fold = (SFS_in_size == SFS_unfolded) && (mse.fold);
	
	double temp;
	if(fold && SFS_size > 2){ temp = double(SFS_in[1]) + double(SFS_in[SFS_unfolded-1]);  }
	else{ temp = SFS_in[1]; }
	double total_snps = temp; 
	out_ptr[1] = temp;
		
	for(int k = 2; k < SFS_size; k++){ 
    	if(fold && k != SFS_unfolded-k){ temp = double(a_ptr[k-2])*(double(SFS_in[k]) + double(SFS_in[SFS_unfolded-k])); } 
    	else { temp = double(a_ptr[k-2])*double(SFS_in[k]); }
    	out_ptr[k] = temp;
    	total_snps += temp;
    } 
	
	out_ptr[0] = 0.f;
	if(zero_class) 
		out_ptr[0] = 1.f - total_snps;
	
	double divisor = 1.f;
	if(!zero_class){
		divisor = total_snps;
		for(int k = 0; k < SFS_size; k++){ out_ptr[k] /= divisor; }  
	}
	
	return SFS_out;
}

py::array_t<double> calc_sfs_mse(py::array_t<const double> & gamma, py::array_t<const double> & dominance, const double F, py::array_t<const double> & proportion, const double theta, py::array_t<const double> & alpha, Spectrum::MSE & mse){
	py::buffer_info buf1 = gamma.request(), buf2 = dominance.request(), buf3 = proportion.request();

    if (buf1.ndim != 1 || buf2.ndim != 1 || buf3.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");

    if (buf1.size != buf2.size || buf2.size != buf3.size)
        throw std::runtime_error("Input shapes must match");
    
    const double * g_ptr = (const double *)buf1.ptr;
    const double * h_ptr = (const double *)buf2.ptr;
    const double * p_ptr = (const double *)buf3.ptr;

    sfs_mse_expectation(mse, g_ptr, h_ptr, F, p_ptr, buf1.size, theta, 1.f);
	
	return normalize_SFS(mse.h_frequency_spectrum, mse.sample_size, alpha, mse);
}

// py::array_t<float> renormalize_SFS(py::array_t<const float> & sfs, py::array_t<const float> & alpha, Spectrum::MSE & mse){
// 	py::buffer_info buf1 = sfs.request();
// 	if (buf1.ndim != 1)
//         throw std::runtime_error("Number of dimensions must be one");
//     
//     int SFS_size = mse.SFS_size;
//     
//     if (buf1.size != SFS_size)
//         throw std::runtime_error("Input shapes must match");
//     
//     const float * in_ptr = (const float *)buf1.ptr;
//     
//     return normalize_SFS(in_ptr, SFS_size, alpha, mse);
// }

PYBIND11_MODULE(mse_gpu, m)
{	
	py::class_<Spectrum::MSE>(m, "MSE")
        .def(py::init<const int, const int, const bool, const bool, int>(),
             py::arg("sample_size"), 
             py::arg("eff_num_chromosomes"),
             py::arg("fold"),
             py::arg("zero_class"),
             py::arg("cuda_device")=-1);

	m.def("calc_sfs_mse", &calc_sfs_mse);
	//m.def("renormalize_SFS", &renormalize_SFS);
}