#include "spectrum.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void sfs_mse_expectation(Spectrum::MSE & mse, const float * gamma, const float * h, const float F, const float * proportion, const int num_categories, const float theta, const float num_sites);

py::array_t<float> normalize_SFS(const float * SFS_in, const int SFS_in_size, py::array_t<const float> & alpha, Spectrum::MSE & mse){	
	py::buffer_info buf = alpha.request();
    
    if (buf.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");
    
    int SFS_size = mse.SFS_size;
    int SFS_unfolded =  mse.sample_size;
    bool zero_class = mse.zero_class;
        
    if (buf.size != SFS_size - 2)
    	throw std::runtime_error("Alpha size must be 2 less than SFS_size");
    
    const float * a_ptr = (const float *)buf.ptr;
    	
    auto SFS_out = py::array_t<float>(SFS_size);
	py::buffer_info buf2 = SFS_out.request();
	float * out_ptr = (float *)buf2.ptr;
    
	bool fold = (SFS_in_size == SFS_unfolded) && (mse.fold);
	
	float temp;
	if(fold && SFS_size > 2){ temp = SFS_in[1] + SFS_in[SFS_unfolded-1];  }
	else{ temp = SFS_in[1]; }
	double total_snps = temp; 
	out_ptr[1] = temp;
		
	for(int k = 2; k < SFS_size; k++){ 
    	if(fold && k != SFS_unfolded-k){ temp = a_ptr[k-2]*(SFS_in[k] + SFS_in[SFS_unfolded-k]); } 
    	else { temp = a_ptr[k-2]*SFS_in[k]; }
    	out_ptr[k] = temp;
    	total_snps += temp;
    } 
	
	out_ptr[0] = 0.f;
	if(zero_class) 
		out_ptr[0] = 1.f - total_snps;
	
	float divisor = 1.f;
	if(!zero_class){
		divisor = total_snps;
		for(int k = 0; k < SFS_size; k++){ out_ptr[k] /= divisor; }  
	}
	
	return SFS_out;
}

py::array_t<float> calc_sfs_mse(py::array_t<const float> & gamma, py::array_t<const float> & dominance, const float F, py::array_t<const float> & proportion, const float theta, py::array_t<const float> & alpha, Spectrum::MSE & mse){
	py::buffer_info buf1 = gamma.request(), buf2 = dominance.request(), buf3 = proportion.request();

    if (buf1.ndim != 1 || buf2.ndim != 1 || buf3.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");

    if (buf1.size != buf2.size || buf2.size != buf3.size)
        throw std::runtime_error("Input shapes must match");
    
    const float * g_ptr = (const float *)buf1.ptr;
    const float * h_ptr = (const float *)buf2.ptr;
    const float * p_ptr = (const float *)buf3.ptr;

    sfs_mse_expectation(mse, g_ptr, h_ptr, F, p_ptr, buf1.size, theta, 1.f);
	
	return normalize_SFS(mse.h_frequency_spectrum, mse.sample_size, alpha, mse);
}

py::array_t<float> renormalize_SFS(py::array_t<const float> & sfs, py::array_t<const float> & alpha, Spectrum::MSE & mse){
	py::buffer_info buf1 = sfs.request();
	if (buf1.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");
    
    int SFS_size = mse.SFS_size;
    
    if (buf1.size != SFS_size)
        throw std::runtime_error("Input shapes must match");
    
    const float * in_ptr = (const float *)buf1.ptr;
    
    return normalize_SFS(in_ptr, SFS_size, alpha, mse);
}

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
	m.def("renormalize_SFS", &renormalize_SFS);
}