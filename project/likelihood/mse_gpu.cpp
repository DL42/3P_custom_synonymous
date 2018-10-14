#include "spectrum.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void sfs_mse_expectation(Spectrum::MSE & mse, float * gamma, float * sel_prop, int num_categories, float theta, float num_sites);

py::array_t<float> calc_sfs_mse(py::array_t<float> gamma, py::array_t<float> sel_prop, float theta, float num_sites, bool fold, bool zero_class, Spectrum::MSE & mse){
	py::buffer_info buf1 = gamma.request(), buf2 = sel_prop.request();

    if (buf1.ndim != 1 || buf2.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");

    if (buf1.size != buf2.size)
        throw std::runtime_error("Input shapes must match");
    
    float * g_ptr = (float *)buf1.ptr;
    float * p_ptr = (float *)buf2.ptr;
    
    sfs_mse_expectation(mse, g_ptr, p_ptr, buf1.size, theta, num_sites);
    
    int SFS_size = mse.sample_size;
    int SFS_unfolded = SFS_size;
    if(fold){ SFS_size = ((SFS_size%2)+SFS_size)/2 + 1; }
   
	auto SFS_DFE = py::array_t<float>(SFS_size);
	py::buffer_info buf3 = SFS_DFE.request();
	float * sfs_ptr = (float *)buf3.ptr;
	
	sfs_ptr[0] = 0;
	if(zero_class) 
		sfs_ptr[0] = mse.h_frequency_spectrum[0]/num_sites;
	
	float divisor = num_sites;
	if(!zero_class)
		divisor = num_sites - mse.h_frequency_spectrum[0];
		
	if(fold){
    	for(int k = 1; k < SFS_size; k++){ 
    		if(k != SFS_unfolded-k){ sfs_ptr[k] = (mse.h_frequency_spectrum[k] + mse.h_frequency_spectrum[SFS_unfolded-k])/divisor; } 
    		else { sfs_ptr[k] = mse.h_frequency_spectrum[k]/divisor; } 
    	} 
	}else{
		for(int k = 1; k < SFS_size; k++)
    		sfs_ptr[k] = mse.h_frequency_spectrum[k]/divisor; 
	}
	
	return SFS_DFE;
}

PYBIND11_MODULE(mse_gpu, m)
{	
	py::class_<Spectrum::MSE>(m, "MSE")
        .def(py::init<const int, const int, const float, int>(),
             py::arg("sample_size"), 
             py::arg("population_size"),
             py::arg("inbreeding"),
             py::arg("cuda_device")=-1);

	m.def("calc_sfs_mse", &calc_sfs_mse);
}