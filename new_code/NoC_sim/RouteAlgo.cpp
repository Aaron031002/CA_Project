#include "RouteAlgo.h"

RouteAlgo::RouteAlgo(int mesh_width, int mesh_height)
    : route_name_("XY"), mesh_width_(mesh_width),
      mesh_height_(mesh_height < 0 ? mesh_width : mesh_height) {}
RouteAlgo::RouteAlgo(std::string route_name, int mesh_width, int mesh_height)
    : route_name_(route_name), mesh_width_(mesh_width),
      mesh_height_(mesh_height < 0 ? mesh_width : mesh_height) {}

// user APIs
std::pair<LLM_SIM_NOC::PathUnion_t, LLM_SIM_NOC::AllPath_t> RouteAlgo::get_path(const int& src_id, const int& dst_id) { // for unicast
    if (route_name_ == "TORUS_XY")
        return torus_xy_route(src_id, {dst_id});
    return xy_route(src_id, {dst_id}); // default: XY
}
std::pair<LLM_SIM_NOC::PathUnion_t, LLM_SIM_NOC::AllPath_t> RouteAlgo::get_path(const int& src_id, const std::vector<int>& dst_ids) { // for multicast
    if (route_name_ == "TORUS_XY")
        return torus_xy_route(src_id, dst_ids);
    return xy_route(src_id, dst_ids); // default: XY
}

// routing algorithms
// ── Torus XY routing ─────────────────────────────────────────────────────────
// In each dimension, always take the shorter of the two paths (direct vs
// wrap-around). The direct distance is |d|, and the wrap-around distance is
// (dimension size) - |d|; whichever is smaller is chosen. X uses mesh_width_
// and Y uses mesh_height_ independently, so a rectangular MxN torus wraps
// correctly on both axes, not just a square one.
std::pair<LLM_SIM_NOC::PathUnion_t, LLM_SIM_NOC::AllPath_t>
RouteAlgo::torus_xy_route(const int& src_id, const std::vector<int>& dst_ids) {
    using namespace LLM_SIM_NOC;
    PathUnion_t paths;
    AllPath_t   all_unicast_paths;

    for (const int dst_id : dst_ids) {
        Path_t unicast_path;
        auto cur = id2coord(src_id);
        const auto dst = id2coord(dst_id);
        int curr_id = src_id;

        // ── X dimension ──────────────────────────────────────────────────────
        while (cur.first != dst.first) {
            int dx      = dst.first - cur.first;
            int direct  = std::abs(dx);
            int wrap    = mesh_width_ - direct;
            bool go_direct = (direct <= wrap);
            // go_direct → step toward dst; otherwise step away (wrap)
            int step = go_direct ? (dx > 0 ? 1 : -1) : (dx > 0 ? -1 : 1);
            DIR dir  = (step > 0) ? DIR::E : DIR::W;
            cur.first = (cur.first + step + mesh_width_) % mesh_width_;
            curr_id   = coord2id(cur);
            unicast_path.push_back({curr_id, dir});
        }

        // ── Y dimension ──────────────────────────────────────────────────────
        while (cur.second != dst.second) {
            int dy      = dst.second - cur.second;
            int direct  = std::abs(dy);
            int wrap    = mesh_height_ - direct;
            bool go_direct = (direct <= wrap);
            int step = go_direct ? (dy > 0 ? 1 : -1) : (dy > 0 ? -1 : 1);
            DIR dir  = (step > 0) ? DIR::S : DIR::N;
            cur.second = (cur.second + step + mesh_height_) % mesh_height_;
            curr_id    = coord2id(cur);
            unicast_path.push_back({curr_id, dir});
        }

        if (all_unicast_paths.find(dst_id) == all_unicast_paths.end())
            all_unicast_paths[dst_id] = unicast_path;
        path_union(paths, unicast_path);
    }
    return {paths, all_unicast_paths};
}

// ── Standard mesh XY routing ─────────────────────────────────────────────────
// routing algorithms
std::pair<LLM_SIM_NOC::PathUnion_t, LLM_SIM_NOC::AllPath_t> RouteAlgo::xy_route(const int& src_id, const std::vector<int>& dst_ids) {
    LLM_SIM_NOC::PathUnion_t paths;
    LLM_SIM_NOC::AllPath_t all_unicast_paths;
    for(size_t dst_idx = 0; dst_idx < dst_ids.size(); dst_idx++) {
        const int dst_id = dst_ids[dst_idx];
        // unicast xy routing
        LLM_SIM_NOC::Path_t unicast_path;
        auto src_coor = id2coord(src_id);
        auto dst_coor = id2coord(dst_id);
        int curr_id = src_id;
        
        while(src_coor.first != dst_coor.first) {
            if(src_coor.first < dst_coor.first) {
                // move east
                src_coor.first += 1;
                curr_id = coord2id(src_coor);
                unicast_path.push_back({curr_id, LLM_SIM_NOC::DIR::E});
            }
            else {
                // move west
                src_coor.first -= 1;
                curr_id = coord2id(src_coor);
                unicast_path.push_back({curr_id, LLM_SIM_NOC::DIR::W});
            }
        }
        while(src_coor.second != dst_coor.second) {
            if(src_coor.second < dst_coor.second) {
                // move south
                src_coor.second += 1;
                curr_id = coord2id(src_coor);
                unicast_path.push_back({curr_id, LLM_SIM_NOC::DIR::S});
            }
            else {
                // move north
                src_coor.second -= 1;
                curr_id = coord2id(src_coor);
                unicast_path.push_back({curr_id, LLM_SIM_NOC::DIR::N});
            }
        }
        if(all_unicast_paths.find(dst_id) == all_unicast_paths.end()) {
            all_unicast_paths[dst_id] = unicast_path;
        }
        path_union(paths, unicast_path);
    } 

    return {paths, all_unicast_paths};
}


void RouteAlgo::path_union(LLM_SIM_NOC::PathUnion_t& paths, const LLM_SIM_NOC::Path_t& unicast_path) const {
    // check if the path is in the paths
        for(size_t hop_idx = 0; hop_idx < unicast_path.size(); hop_idx++) {
            int route_id = unicast_path[hop_idx].curr_id;
            LLM_SIM_NOC::DIR dir = unicast_path[hop_idx].dir;

            if(paths.find(route_id) != paths.end()) { // the router id is already recorded
                auto dir_dist = paths[route_id].dir_dist;
                bool dir_found = false;
                for(size_t dd_idx = 0; dd_idx < dir_dist.size(); dd_idx++) {
                    if(dir_dist[dd_idx].first == dir) { // the direction is already recorded
                        if(dir_dist[dd_idx].second > hop_idx + 1)
                            dir_dist[dd_idx].second = (hop_idx + 1);
                        paths[route_id].dir_dist = dir_dist;
                        dir_found = true;
                        break;
                    }
                }
                if(!dir_found)
                    paths[route_id].dir_dist.push_back({dir, (hop_idx + 1)});
            }
            else { // The router is not recorded yet
                std::vector<std::pair<LLM_SIM_NOC::DIR, uint64_t>> dir_dist;
                dir_dist.push_back({dir, (hop_idx + 1)});
                LLM_SIM_NOC::UnionHop_t path;
                path.curr_id = route_id;
                path.dir_dist = dir_dist;
                paths[route_id] = path;
            }
        }
}


// helper methods
inline std::pair<int, int> RouteAlgo::id2coord(const int& id) const {
    int x = id % mesh_width_;
    int y = id / mesh_width_;
    return {x, y};
}
inline int RouteAlgo::coord2id(const std::pair<int, int>& coord) const {
    return coord.second * mesh_width_ + coord.first;
}