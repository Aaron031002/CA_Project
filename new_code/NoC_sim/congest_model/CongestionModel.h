#ifndef LLM_SIM_CONGESTION_MODEL_H
#define LLM_SIM_CONGESTION_MODEL_H

#include "noc_utils.h"
#include "noc_param.h"
#include <systemc.h>
#include <vector>
#include <map>

namespace LLM_SIM_NOC {

class CongestionModel : public sc_module {
public:
    SC_HAS_PROCESS(CongestionModel);
    
    CongestionModel(sc_module_name name) : sc_module(name) {
        // Initialize congestion matrix for all nodes
        congestion_matrix.resize(NODE_NUM);
        for (int i = 0; i < NODE_NUM; ++i) {
            congestion_matrix[i].resize(5); // 4 directions + local
        }
        
        // Initialize link utilization tracking
        link_utilization.resize(NODE_NUM);
        for (int i = 0; i < NODE_NUM; ++i) {
            link_utilization[i].resize(5, 0.0);
        }
        
        SC_METHOD(update_congestion);
        sensitive << clk.pos();
    }
    
    // Ports
    sc_in<bool> clk;
    
    // Methods
    void update_link_utilization(int node_id, DIR direction, bool active);
    double get_link_congestion(int node_id, DIR direction);
    CongestionInfo get_node_congestion(int node_id);
    void update_congestion();
    bool is_link_congested(int node_id, DIR direction);
    std::vector<DIR> get_available_directions(int node_id, std::pair<int,int> dst_coord);
    
private:
    CongestionMatrix congestion_matrix;
    std::vector<std::vector<double>> link_utilization;
    std::map<int, int> packet_counters;
    
    // Helper methods
    std::pair<int, int> id2coord(int id);
    int coord2id(std::pair<int, int> coord);
    bool is_valid_coord(std::pair<int, int> coord);
};

} // namespace LLM_SIM_NOC

#endif