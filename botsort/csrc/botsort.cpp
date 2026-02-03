#include "include/botsort.h"

#include <optional>
#include <unordered_set>

#include <opencv2/imgproc.hpp>

#include "include/DataType.h"
#include "include/INIReader.h"
#include "include/matching.h"
#include "include/profiler.h"

/*
* bindings:
*   BoTSoRT
*       track
*   Config
*       TrackerParams
*       GMC_Params
*       
*/

namespace
{
template<typename T>
bool requires_load(const Config<T> &config)
{
    return std::holds_alternative<std::string>(config) &&
           !std::get<std::string>(config).empty();
}

template<typename T>
bool not_empty(const Config<T> &config)
{
    bool has_config = !std::holds_alternative<std::monostate>(config);
    bool is_non_empty = !(std::holds_alternative<std::string>(config) &&
                          std::get<std::string>(config).empty());

    return has_config && is_non_empty;
}

template<typename T>
T fetch_config(const Config<T> &config,
               std::function<T(const std::string &)> loader)
{
    if (std::holds_alternative<T>(config))
    {
        return std::get<T>(config);
    }

    if (requires_load(config))
    {
        return loader(std::get<std::string>(config));
    }

    throw std::runtime_error("Config is empty");
}
}// namespace

Botsort::Botsort(const Config<TrackerParams> &tracker_config,
                 const Config<GMC_Params> &gmc_config)
                 //const Config<ReIDParams> &reid_config,
                 //const std::string &reid_onnx_model_path)
{
    auto tracker_params = fetch_config<TrackerParams>
    (tracker_config, TrackerParams::load_config);
    _load_params_from_config(tracker_params);

    // Tracker module
    _frame_id = 0;
    _buffer_size = static_cast<uint8_t>(_frame_rate / 30.0 * _track_buffer);
    _max_time_lost = _buffer_size;
    _kalman_filter = std::make_unique<KalmanFilter>(static_cast<double>(1.0 / _frame_rate));

    // Global motion compensation module
    if (_gmc_enabled && not_empty(gmc_config))
    {
        auto gmc_params = fetch_config<GMC_Params>
        (gmc_config, [this](const std::string &config_path) {
            return GMC_Params::load_config(
                    GlobalMotionCompensation::GMC_method_map[_gmc_method_name],
                    config_path);
        });
        _gmc_algo = std::make_unique<GlobalMotionCompensation>(gmc_params);
    }
    else
    {
        // std::cout << "GMC disabled" << std::endl;
        _gmc_enabled = false;
    }
}

py::array_t<float>
Botsort::track(const py::array_t<float> &box_tlwh,
               const py::array_t<float> &score,
               const py::array_t<int64_t> &class_ids,
               const py::array_t<uint8_t> &frame)
{
    py::buffer_info buf = frame.request(false);
    py::buffer_info box_tlwh_buf = box_tlwh.request(false);
    py::buffer_info score_buf = score.request(false);
    py::buffer_info class_ids_buf = class_ids.request(false);

    if (buf.shape.size() != 3)
    {
        throw std::length_error("must pass a single frame to track. Batch not supported");
    }
    if (std::min({buf.shape[0], buf.shape[1], buf.shape[2]}) != buf.shape[2]){
        throw std::runtime_error("Final dim Channels must be smallest");
    }
    if (box_tlwh.ndim() != 1) {
        throw std::runtime_error("box array must be 1-dimensional");
    }
    if (score.ndim() != 1) {
        throw std::runtime_error("score array must be 1-dimensional");
    }
    if (class_ids.ndim() != 1) {
        throw std::runtime_error("class ids array must be 1-dimensional");
    }

    int rows = buf.shape[0];
    int cols = buf.shape[1];

    std::span<const float> box_tlwh_span{static_cast<float*> (box_tlwh_buf.ptr), 
                                         static_cast<size_t> (box_tlwh_buf.size)};
    std::span<const float> score_span{static_cast<float*> (score_buf.ptr),
                                      static_cast<size_t> (score_buf.size)};
    std::span<const int64_t> class_ids_span{static_cast<int64_t*> (class_ids_buf.ptr), 
                                            static_cast<size_t> (class_ids_buf.size)};
    
    cv::Mat cv_frame(rows, cols, CV_8UC3, static_cast<int64_t*>(buf.ptr));
    
    std::vector<std::shared_ptr<Track>> output_tracks;
    Botsort::track(
        box_tlwh_span, score_span, class_ids_span, cv_frame, output_tracks);
    py::array_t<float> output({static_cast<int>(output_tracks.size()), 7});
    auto view = output.mutable_unchecked<2>();

    for (auto i = 0; i < static_cast<int>(output_tracks.size()); ++i){
        view(i, 0) = output_tracks[i]->get_tlwh()[0];
        view(i, 1) = output_tracks[i]->get_tlwh()[1];
        view(i, 2) = output_tracks[i]->get_tlwh()[2];
        view(i, 3) = output_tracks[i]->get_tlwh()[3];
        view(i, 4) = static_cast<float>(output_tracks[i]->get_class_id());
        view(i, 5) = output_tracks[i]->get_score();
        view(i, 6) = static_cast<float>(output_tracks[i]->track_id);
    }
    return output;
}

void
Botsort::track(std::span<const float> &box_tlwh,
               std::span<const float> &score,
               std::span<const int64_t> &class_ids,
               const cv::Mat &frame,
               std::vector<std::shared_ptr<Track>> &output_tracks)
{
    //PROFILE_FUNCTION();
    ////////////////// CREATE TRACK OBJECT FOR ALL THE DETECTIONS //////////////////
    // For all detections, extract features, create tracks and classify on the segregate of confidence

    _frame_id++;
    std::vector<std::shared_ptr<Track>> activated_tracks, refind_tracks;
    std::vector<std::shared_ptr<Track>> detections_high_conf, detections_low_conf;
    
    detections_low_conf.reserve(class_ids.size());
    detections_high_conf.reserve(class_ids.size());

    for (size_t i = 0; i < box_tlwh.size(); i += 4)
    {
        float x = std::max(0.0f, box_tlwh[0 + i]);
        float y = std::max(0.0f, box_tlwh[1 + i]);
        float width = std::min(static_cast<float>(frame.cols - 1),
                                box_tlwh[2 + i]);
        float height = std::min(static_cast<float>(frame.rows - 1),
                                box_tlwh[3 + i]);

        std::shared_ptr<Track> tracklet;
        std::vector<float> tlwh = {x, y, width, height};

        if (score[i/4] > _track_low_thresh)
        {
            tracklet = std::make_shared<Track>(
                        tlwh, score[i/4], class_ids[i/4]);
            
            //// std::cout << i << std::endl;
            //// std::cout << score[i/4] << std::endl;
            //// std::cout << class_ids[i/4] << std::endl;
            if (score[i/4] >= _track_high_thresh){
                
                detections_high_conf.push_back(tracklet);
            } else {
                detections_low_conf.push_back(tracklet);
            }
        }
    }
    // std::cout << "Low Conf " << detections_low_conf.size() << std::endl;
    // std::cout << "High Conf " << detections_high_conf.size() << std::endl;

    // Segregate tracks in unconfirmed and tracked tracks
    std::vector<std::shared_ptr<Track>> unconfirmed_tracks, tracked_tracks;
    for (const std::shared_ptr<Track> &track: _tracked_tracks)
    {
        if (track->is_activated)
        {
            tracked_tracks.push_back(track);
        }
        else
        {
            unconfirmed_tracks.push_back(track);
        }
    }

    
    // std::cout << "Confirmed " << tracked_tracks.size() << std::endl;
    // std::cout << "Unconfirmed " << unconfirmed_tracks.size() << std::endl;

    ////////////////// CREATE TRACK OBJECT FOR ALL THE DETECTIONS //////////////////


    ////////////////// Apply KF predict and GMC before running association algorithm //////////////////
    // Merge currently tracked tracks and lost tracks
    std::vector<std::shared_ptr<Track>> tracks_pool;
    tracks_pool = _merge_track_lists(tracked_tracks, _lost_tracks);

    // Predict the location of the tracks with KF (even for lost tracks)
    Track::multi_predict(tracks_pool, *_kalman_filter);

    // Estimate camera motion and apply camera motion compensation
    if (_gmc_enabled)
    {
        HomographyMatrix H = _gmc_algo->apply(frame, box_tlwh);
        Track::multi_gmc(tracks_pool, H);
        Track::multi_gmc(unconfirmed_tracks, H);
    }
    ////////////////// Apply KF predict and GMC before running association algorithm //////////////////


    ////////////////// ASSOCIATION ALGORITHM STARTS HERE //////////////////
    ////////////////// First association, with high score detection boxes //////////////////
    // Find IoU distance between all tracked tracks and high confidence detections
    CostMatrix iou_dists, raw_emd_dist, iou_dists_mask_1st_association,
            emd_dist_mask_1st_association;

    std::tie(iou_dists, iou_dists_mask_1st_association) =
            iou_distance(tracks_pool, detections_high_conf, _proximity_thresh);

    fuse_score(iou_dists, detections_high_conf);// Fuse the score with IoU distance

    // Fuse the IoU distance and embedding distance to get the final distance matrix
    CostMatrix distances_first_association = fuse_iou_with_emb(
        iou_dists,
        raw_emd_dist,
        iou_dists_mask_1st_association,
        emd_dist_mask_1st_association);

    // Perform linear assignment on the final distance matrix, LAPJV algorithm is used here
    AssociationData first_associations = linear_assignment(
        distances_first_association,
        _match_thresh);
    
    // Update the tracks with the associated detections
    for (const std::pair<int, int> &match: first_associations.matches)
    {
        const std::shared_ptr<Track> &track = tracks_pool[match.first];
        const std::shared_ptr<Track> &detection = detections_high_conf[match.second];
        
        // If track was being actively tracked, we update the track with the new associated detection
        if (track->state == TrackState::Tracked)
        {
            track->update(*_kalman_filter, *detection, _frame_id);
            activated_tracks.push_back(track);
        }
        else
        {
            // If track was not being actively tracked, we re-activate the track with the new associated detection
            // NOTE: There should be a minimum number of frames before a track is re-activated
            track->re_activate(*_kalman_filter, *detection, _frame_id, track_count, false);
            refind_tracks.push_back(track);
        }
    }
    
    // std::cout << "Activated Tracks " << detections_low_conf.size() << std::endl;
    // std::cout << "Refind Tracks " << detections_low_conf.size() << std::endl;
    ////////////////// First association, with high score detection boxes //////////////////


    ////////////////// Second association, with low score detection boxes //////////////////
    // Get all unmatched but tracked tracks after the first association, these tracks will be used for the second association
    std::vector<std::shared_ptr<Track>> unmatched_tracks_after_1st_association;
    for (int track_idx: first_associations.unmatched_track_indices)
    {
        const std::shared_ptr<Track> &track = tracks_pool[track_idx];
        if (track->state == TrackState::Tracked)
        {
            unmatched_tracks_after_1st_association.push_back(track);
        }
    }

    // Find IoU distance between unmatched but tracked tracks left after the first association and low confidence detections
    CostMatrix iou_dists_second;
    iou_dists_second = iou_distance(unmatched_tracks_after_1st_association,
                                    detections_low_conf);

    // Perform linear assignment on the distance matrix, LAPJV algorithm is used here
    AssociationData second_associations = linear_assignment(iou_dists_second, 0.5);

    // Update the tracks with the associated detections
    for (const std::pair<int, int> &match: second_associations.matches)
    {
        const std::shared_ptr<Track> &track =
                unmatched_tracks_after_1st_association[match.first];
        const std::shared_ptr<Track> &detection =
                detections_low_conf[match.second];

        // If track was being actively tracked, we update the track with the new associated detection
        if (track->state == TrackState::Tracked) {
            track->update(*_kalman_filter, *detection, _frame_id);
            activated_tracks.push_back(track);
        } else {
            // If track was not being actively tracked, we re-activate the track with the new associated detection
            // NOTE: There should be a minimum number of frames before a track is re-activated
            track->re_activate(*_kalman_filter, *detection, _frame_id, track_count, false);
            refind_tracks.push_back(track);
        }
    }

    // std::cout << "Activated Tracks 2nd " << detections_low_conf.size() << std::endl;
    // std::cout << "Refind Tracks 2nd " << detections_low_conf.size() << std::endl;
    // The tracks that are not associated with any detection even after the second association are marked as lost
    std::vector<std::shared_ptr<Track>> lost_tracks;
    for (int unmatched_track_index: second_associations.unmatched_track_indices)
    {
        const std::shared_ptr<Track> &track =
                unmatched_tracks_after_1st_association[unmatched_track_index];
        if (track->state != TrackState::Lost)
        {
            track->mark_lost();
            lost_tracks.push_back(track);
        }
    }
    
    // std::cout << "Lost Tracks " << lost_tracks.size() << std::endl;
    ////////////////// Second association, with low score detection boxes //////////////////


    ////////////////// Deal with unconfirmed tracks //////////////////
    std::vector<std::shared_ptr<Track>> unmatched_detections_after_1st_association;
    for (int detection_idx: first_associations.unmatched_det_indices)
    {
        const std::shared_ptr<Track> &detection =
                detections_high_conf[detection_idx];
        unmatched_detections_after_1st_association.push_back(detection);
    }
    
    // std::cout << "Second Association Unmatched " << unmatched_detections_after_1st_association.size() << std::endl;
    //Find IoU distance between unconfirmed tracks and high confidence detections left after the first association
    CostMatrix iou_dists_unconfirmed, raw_emd_dist_unconfirmed,
            iou_dists_mask_unconfirmed, emd_dist_mask_unconfirmed;

    std::tie(iou_dists_unconfirmed, iou_dists_mask_unconfirmed) = iou_distance(
            unconfirmed_tracks, unmatched_detections_after_1st_association,
            _proximity_thresh);
    fuse_score(iou_dists_unconfirmed, unmatched_detections_after_1st_association);

    // Fuse the IoU distance and the embedding distance
    CostMatrix distances_unconfirmed = fuse_iou_with_emb(
            iou_dists_unconfirmed, raw_emd_dist_unconfirmed,
            iou_dists_mask_unconfirmed, emd_dist_mask_unconfirmed);

    // Perform linear assignment on the distance matrix, LAPJV algorithm is used here
    AssociationData unconfirmed_associations =
            linear_assignment(distances_unconfirmed, 0.7);

    for (const std::pair<int, int> &match: unconfirmed_associations.matches)
    {
        const std::shared_ptr<Track> &track = unconfirmed_tracks[match.first];
        const std::shared_ptr<Track> &detection =
                unmatched_detections_after_1st_association[match.second];

        // If the unconfirmed track is associated with a detection we update the track with the new associated detection
        // and add the track to the activated tracks list
        track->update(*_kalman_filter, *detection, _frame_id);
        activated_tracks.push_back(track);
    }
    
    // // std::cout << "Activated Tracks before removal: " << activated_tracks.size() << std::endl;

    // All the unconfirmed tracks that are not associated with any detection are marked as removed
    std::vector<std::shared_ptr<Track>> removed_tracks;
    for (int unmatched_track_index: unconfirmed_associations.unmatched_track_indices)
    {
        const std::shared_ptr<Track> &track =
                unconfirmed_tracks[unmatched_track_index];
        track->mark_removed();
        removed_tracks.push_back(track);
    }
    
    //// std::cout << "Removed Tracks " << removed_tracks.size() << std::endl;
    ////////////////// Deal with unconfirmed tracks //////////////////

    ////////////////// Initialize new tracks //////////////////
    std::vector<std::shared_ptr<Track>> unmatched_high_conf_detections;
    for (int detection_idx: unconfirmed_associations.unmatched_det_indices)
    {
        const std::shared_ptr<Track> &detection =
                unmatched_detections_after_1st_association[detection_idx];
        if (detection->get_score() >= _new_track_thresh)
        {
            /*auto tlwh = detection->get_tlwh();
            std::cout << "New Track" << std::endl;
            for (auto point: tlwh)
                {std::cout << point << ' ';}
            std::cout << std::endl; */

            detection->activate(*_kalman_filter, _frame_id, track_count);            
            activated_tracks.push_back(detection);
        }
    }
    
    // std::cout << "Activated Tracks New " << detections_low_conf.size() << std::endl;
    ////////////////// Initialize new tracks //////////////////


    ////////////////// Update lost tracks state //////////////////
    for (const std::shared_ptr<Track> &track: _lost_tracks)
    {
        if (_frame_id - track->end_frame() > _max_time_lost)
        {
            
            // std::cout << "Removed: " << track->track_id << std::endl;
            track->mark_removed();
            removed_tracks.push_back(track);
        }
    }
    ////////////////// Update lost tracks state //////////////////


    ////////////////// Clean up the track lists //////////////////
    std::vector<std::shared_ptr<Track>> updated_tracked_tracks;
    for (const std::shared_ptr<Track> &_tracked_track: _tracked_tracks)
    {
        if (_tracked_track->state == TrackState::Tracked)
        {
            updated_tracked_tracks.push_back(_tracked_track);
        }
    }
    _tracked_tracks =
            _merge_track_lists(updated_tracked_tracks, activated_tracks);
    _tracked_tracks = _merge_track_lists(_tracked_tracks, refind_tracks);

    _lost_tracks = _merge_track_lists(_lost_tracks, lost_tracks);
    _lost_tracks = _remove_from_list(_lost_tracks, _tracked_tracks);
    _lost_tracks = _remove_from_list(_lost_tracks, removed_tracks);

    std::vector<std::shared_ptr<Track>> tracked_tracks_cleaned,
            lost_tracks_cleaned;
    _remove_duplicate_tracks(tracked_tracks_cleaned, lost_tracks_cleaned,
                             _tracked_tracks, _lost_tracks);
    _tracked_tracks = tracked_tracks_cleaned;
    _lost_tracks = lost_tracks_cleaned;
    ////////////////// Clean up the track lists //////////////////


    ////////////////// Update output tracks //////////////////
    for (const std::shared_ptr<Track> &track: _tracked_tracks)
    {
        if (track->is_activated)
        {
            output_tracks.push_back(track);
        }
    }

    return;
}

std::vector<std::shared_ptr<Track>>
Botsort::_merge_track_lists(std::vector<std::shared_ptr<Track>> &tracks_list_a,
                            std::vector<std::shared_ptr<Track>> &tracks_list_b)
{
    std::unordered_set<int> exists;
    std::vector<std::shared_ptr<Track>> merged_tracks_list;

    for (const std::shared_ptr<Track> &track: tracks_list_a)
    {
        exists.insert(track->track_id);
        merged_tracks_list.push_back(track);
    }

    for (const std::shared_ptr<Track> &track: tracks_list_b)
    {
        if (exists.find(track->track_id) == exists.end())
        {
            exists.insert(track->track_id);
            merged_tracks_list.push_back(track);
        }
    }

    return merged_tracks_list;
}


std::vector<std::shared_ptr<Track>> Botsort::_remove_from_list(
        std::vector<std::shared_ptr<Track>> &tracks_list,
        std::vector<std::shared_ptr<Track>> &tracks_to_remove)
{
    std::unordered_set<int> exists;
    std::vector<std::shared_ptr<Track>> new_tracks_list;

    for (const std::shared_ptr<Track> &track: tracks_to_remove)
    {
        exists.insert(track->track_id);
    }

    for (const std::shared_ptr<Track> &track: tracks_list)
    {
        if (exists.find(track->track_id) == exists.end())
        {
            new_tracks_list.push_back(track);
        }
    }

    return new_tracks_list;
}


void Botsort::_remove_duplicate_tracks(
        std::vector<std::shared_ptr<Track>> &result_tracks_a,
        std::vector<std::shared_ptr<Track>> &result_tracks_b,
        std::vector<std::shared_ptr<Track>> &tracks_list_a,
        std::vector<std::shared_ptr<Track>> &tracks_list_b)
{
    CostMatrix iou_dists = iou_distance(tracks_list_a, tracks_list_b);

    std::unordered_set<size_t> dup_a, dup_b;
    for (Eigen::Index i = 0; i < iou_dists.rows(); i++)
    {
        for (Eigen::Index j = 0; j < iou_dists.cols(); j++)
        {
            if (iou_dists(i, j) < 0.15)
            {
                int time_a = static_cast<int>(tracks_list_a[i]->frame_id -
                                              tracks_list_a[i]->start_frame);
                int time_b = static_cast<int>(tracks_list_b[j]->frame_id -
                                              tracks_list_b[j]->start_frame);

                // We make an assumption that the longer trajectory is the correct one
                if (time_a > time_b)
                {
                    // In list b, track with index j is a duplicate
                    dup_b.insert(j);
                }
                else
                {
                    // In list a, track with index i is a duplicate
                    dup_a.insert(i);
                }
            }
        }
    }

    // Remove duplicates from the lists
    for (size_t i = 0; i < tracks_list_a.size(); i++)
    {
        if (dup_a.find(i) == dup_a.end())
        {
            result_tracks_a.push_back(tracks_list_a[i]);
        }
    }

    for (size_t i = 0; i < tracks_list_b.size(); i++)
    {
        if (dup_b.find(i) == dup_b.end())
        {
            result_tracks_b.push_back(tracks_list_b[i]);
        }
    }
}

void Botsort::_load_params_from_config(const TrackerParams &config)
{
    _reid_enabled = config.reid_enabled;
    _gmc_enabled = config.gmc_enabled;
    _track_high_thresh = config.track_high_thresh;
    _track_low_thresh = config.track_low_thresh;
    _new_track_thresh = config.new_track_thresh;
    _track_buffer = config.track_buffer;
    _match_thresh = config.match_thresh;
    _proximity_thresh = config.proximity_thresh;
    _appearance_thresh = config.appearance_thresh;
    _gmc_method_name = config.gmc_method_name;
    _frame_rate = config.frame_rate;
    _lambda = config.lambda;
}

PYBIND11_MODULE(_botsort, m, py::multiple_interpreters::per_interpreter_gil()) {
    auto config = m.def_submodule("configs", "All Config Classes for BotSort");
    bind_tracker_params(config);
    bind_gmc_configs(config);
    bind_gmc_method_enum(config);

    py::class_<Botsort, py::smart_holder>(m, "BotSort")
        .def(py::init<Config<TrackerParams>, Config<GMC_Params>>())
        .def("track", static_cast<
            py::array_t<float> 
            (Botsort::*) 
                      (const py::array_t<float> &/* bounding boxes */, 
                       const py::array_t<float> &/* scores */,
                       const py::array_t<int64_t> &/*class ids*/,
                       const py::array_t<uint8_t> &) /* frame */
            > (&Botsort::track),
                    "BotSort entry/forward function.\n"
                    "\tParams:"
                    "\t\t Bounding Boxes. 1D numpy float array with size divisible by 4"
                    "\t\t Scores. 1D numpy float array"
                    "\t\t Class Ids. 1D numpy long array"
                    "\tReturns:"
                    "\t\t1D numpy array of IDs in order of class ids input");
}