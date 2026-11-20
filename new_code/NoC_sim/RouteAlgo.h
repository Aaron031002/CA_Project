#ifndef LLM_SIM_ROUTER_ROUTEALGO_H
#define LLM_SIM_ROUTER_ROUTEALGO_H

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "NoC_sim/noc_utils.h"

class RouteAlgo {

    public:
        // mesh_height defaults to mesh_width (square mesh) for every caller
        // that doesn't pass it -- id2coord/coord2id/xy_route only ever need
        // the width (correct for any rectangular mesh); mesh_height_ is
        // consulted only by torus_xy_route's Y-axis wrap-around distance,
        // so a caller using plain "XY" never needs to pass it at all.
        explicit RouteAlgo(int mesh_width = LLM_SIM_NOC::MESH_WIDTH,
                            int mesh_height = -1);
        explicit RouteAlgo(std::string route_name, int mesh_width = LLM_SIM_NOC::MESH_WIDTH,
                            int mesh_height = -1);
        // user APIs
        std::pair<LLM_SIM_NOC::PathUnion_t, LLM_SIM_NOC::AllPath_t> get_path(const int& src_id, const int& dst_id); // for unicast
        std::pair<LLM_SIM_NOC::PathUnion_t, LLM_SIM_NOC::AllPath_t> get_path(const int& src_id, const std::vector<int>& dst_ids); // for multicast

    private:
        const std::string route_name_;
        const int mesh_width_;
        // Y-axis wrap-around modulus for torus_xy_route -- independent of
        // mesh_width_ so a rectangular torus wraps correctly on both axes.
        // Defaults to mesh_width_ (square mesh) when mesh_height < 0.
        const int mesh_height_;
        // routing algorithm
        std::pair<LLM_SIM_NOC::PathUnion_t, LLM_SIM_NOC::AllPath_t> xy_route(const int& src_id, const std::vector<int>& dst_ids);
        std::pair<LLM_SIM_NOC::PathUnion_t, LLM_SIM_NOC::AllPath_t> torus_xy_route(const int& src_id, const std::vector<int>& dst_ids);
        void path_union(LLM_SIM_NOC::PathUnion_t& paths, const LLM_SIM_NOC::Path_t& unicast_path) const;
        // helper methods
        inline std::pair<int, int> id2coord(const int& id) const;
        inline int coord2id(const std::pair<int, int>& coord) const;
};

#endif