#include "include/TrackerParams.h"

#include <iostream>

#include "include/DataType.h"
#include "include/INIReader.h"


TrackerParams TrackerParams::load_config(const std::string &config_path)
{
    TrackerParams config{};

    const std::string tracker_name = "BoTSORT";

    INIReader tracker_config(config_path);
    if (tracker_config.ParseError() < 0)
    {
        std::cout << "Can't load " << config_path << std::endl;
        exit(1);
    }

    tracker_config.LoadBoolean(tracker_name, "enable_reid",
                               config.reid_enabled);
    tracker_config.LoadBoolean(tracker_name, "enable_gmc", config.gmc_enabled);
    tracker_config.LoadFloat(tracker_name, "track_high_thresh",
                             config.track_high_thresh);
    tracker_config.LoadFloat(tracker_name, "track_low_thresh",
                             config.track_low_thresh);
    tracker_config.LoadFloat(tracker_name, "new_track_thresh",
                             config.new_track_thresh);
    tracker_config.LoadInteger(tracker_name, "track_buffer",
                               config.track_buffer);
    tracker_config.LoadFloat(tracker_name, "match_thresh", config.match_thresh);
    tracker_config.LoadFloat(tracker_name, "proximity_thresh",
                             config.proximity_thresh);
    tracker_config.LoadFloat(tracker_name, "appearance_thresh",
                             config.appearance_thresh);
    tracker_config.LoadString(tracker_name, "gmc_method",
                              config.gmc_method_name);
    tracker_config.LoadInteger(tracker_name, "frame_rate", config.frame_rate);
    tracker_config.LoadFloat(tracker_name, "lambda", config.lambda);

    return config;
}

void bind_tracker_params(py::module_ &m){
    // I need a better way than this
    py::class_<TrackerParams>(m, "trackerParams")
        .def(py::init<
            bool,bool,
            float,float,float,
            long,
            float,float,float,
            std::string,
            long,float>(), 
            py::arg("reid_enabled"),
            py::arg("gmc_enabled"),
            py::arg("track_high_thresh"),
            py::arg("track_low_thresh"),
            py::arg("new_track_thresh"),
            py::arg("track_buffer"),
            py::arg("match_thresh"),
            py::arg("proximity_thresh"),
            py::arg("appearance_thresh"),
            py::arg("gmc_method_name"),
            py::arg("frame_rate"),
            py::arg("lambda_"))
        .def_readwrite("reid_enabled", &TrackerParams::reid_enabled)
        .def_readwrite("gmc_enabled", &TrackerParams::gmc_enabled)
        .def_readwrite("track_high_thresh", &TrackerParams::track_high_thresh)
        .def_readwrite("track_low_thresh", &TrackerParams::track_low_thresh)
        .def_readwrite("new_track_thresh", &TrackerParams::new_track_thresh)
        .def_readwrite("track_buffer", &TrackerParams::track_buffer)
        .def_readwrite("match_thresh", &TrackerParams::match_thresh)
        .def_readwrite("proximity_thresh", &TrackerParams::proximity_thresh)
        .def_readwrite("appearance_thresh", &TrackerParams::appearance_thresh)
        .def_readwrite("gmc_method_name", &TrackerParams::gmc_method_name)
        .def_readwrite("frame_rate", &TrackerParams::frame_rate)
        .def_readwrite("lambda_", &TrackerParams::lambda)
        /*/.doc()*/;
}