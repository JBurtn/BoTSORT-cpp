#pragma once

#include <string>
#include <variant>
#include <pybind11/pybind11.h>

namespace py = pybind11;

enum GMC_Method
{
    ORB = 0,
    ECC,
    SparseOptFlow,
    OptFlowModified,
    OpenCV_VideoStab
};

struct ORB_Params
{
    float downscale{2.f};
    float inlier_ratio{0.5f};
    float ransac_conf{.99f};
    long ransac_max_iters{500};
};

struct ECC_Params
{
    float downscale{5.f};
    long max_iterations{100};
    float termination_eps{1e-6};
};

struct SparseOptFlow_Params
{
    long max_corners{1000};
    long block_size{3};
    long ransac_max_iters{500};
    double quality_level{0.01};
    double k{0.04};
    double min_distance{1.0};
    float downscale{2.0f};
    float inlier_ratio{0.5f};
    float ransac_conf{0.99f};
    bool use_harris_detector{false};
};

struct OptFlowModified_Params
{
    float downscale{2.0f};
};

struct OpenCV_VideoStab_GMC_Params
{
    float downscale{2.0f};
    float num_features{4000};
    bool detection_masking{true};
};

struct GMC_Params
{
    using MethodParams =
            std::variant<ORB_Params, ECC_Params, SparseOptFlow_Params,
                         OptFlowModified_Params, OpenCV_VideoStab_GMC_Params,
                         std::monostate>;
    
    GMC_Method method_;
    MethodParams method_params_;

    static GMC_Params load_config(GMC_Method method,
                                  const std::string &config_path);
};

inline void bind_gmc_method_enum(py::module_ &m){
    py::enum_<GMC_Method>(m, "GMC_METHOD")
        .value("ORB", ORB)
        .value("ECC", ECC)
        .value("SparseOptFlow", SparseOptFlow)
        .value("OpenCV_VideoStab", OpenCV_VideoStab)
        .export_values();
}

inline void bind_gmc_configs(py::module_ &m){
    py::class_<ORB_Params>(m, "ORBConfig")
        .def(py::init<float, float, float, long>(),
            py::arg("downscale"), py::arg("inlier_ratio"),
            py::arg("ransac_conf"), py::arg("max_iterations"))
        .def_readwrite("downscale", &ORB_Params::downscale)
        .def_readwrite("inlier_ratio", &ORB_Params::inlier_ratio)
        .def_readwrite("ransac_conf", &ORB_Params::ransac_conf)
        .def_readwrite("max_iterations", &ORB_Params::ransac_max_iters);
    
    py::class_<ECC_Params>(m, "ECCConfig")
        .def(py::init<float, long, float>(),
            py::arg("downscale"), py::arg("max_iterations"),py::arg("termination_eps"))
        .def_readwrite("downscale", &ECC_Params::downscale)
        .def_readwrite("max_iterations", &ECC_Params::max_iterations)
        .def_readwrite("termination_eps", &ECC_Params::termination_eps);

    py::class_<SparseOptFlow_Params>(m, "SparseOptFlowConfig")
        .def(py::init<long, long, long, double, double, double, float, float, float, bool>(),
    py::arg("max_corners"), py::arg("block_size"), py::arg("ransac_max_iters"),py::arg("quality_level"),
    py::arg("k"), py::arg("min_distance"), py::arg("downscale"), py::arg("inlier_ratio"), py::arg("ransac_conf"), py::arg("use_harris_detector"))
        .def_readwrite("downscale", &SparseOptFlow_Params::downscale)
        .def_readwrite("block_size", &SparseOptFlow_Params::block_size)
        .def_readwrite("ransac_max_iters", &SparseOptFlow_Params::ransac_max_iters)
        .def_readwrite("quality_level", &SparseOptFlow_Params::quality_level)
        .def_readwrite("k", &SparseOptFlow_Params::k)
        .def_readwrite("min_distance", &SparseOptFlow_Params::min_distance)
        .def_readwrite("max_corners", &SparseOptFlow_Params::max_corners)
        .def_readwrite("inlier_ratio", &SparseOptFlow_Params::inlier_ratio)
        .def_readwrite("ransac_conf", &SparseOptFlow_Params::ransac_conf)
        .def_readwrite("use_harris_dectector", &SparseOptFlow_Params::use_harris_detector);
    
    py::class_<OpenCV_VideoStab_GMC_Params>(m, "OpenCVVideoStabGMCConfig")
        .def(py::init<float, float, bool>(),
            py::arg("downscale"), py::arg("num_features"), py::arg("detection_masking"))
        .def_readwrite("downscale", &OpenCV_VideoStab_GMC_Params::downscale)
        .def_readwrite("num_features", &OpenCV_VideoStab_GMC_Params::num_features)
        .def_readwrite("detection_masking", &OpenCV_VideoStab_GMC_Params::detection_masking);

    py::class_<GMC_Params>(m, "GMC_Config")
        .def(py::init<GMC_Method, GMC_Params::MethodParams>(),
             py::arg("method"), py::arg("method_params"))
        .def_readonly("method", &GMC_Params::method_)
        .def_readwrite("method_params", &GMC_Params::method_params_);
}
    