#include "CongestionModel.h"

namespace LLM_SIM_NOC {

std::pair<int, int> CongestionModel::id2coord(int id) {
    return std::make_pair(id % NOC_WIDTH, id / NOC_WIDTH);
}

int CongestionModel::coord2id(std::pair<int, int> coord) {
    return coord.second * NOC_WIDTH + coord.first;
}

bool CongestionModel::is_valid_coord(std::pair<int, int> coord) {
    return coord.first >= 0 && coord.first < NOC_WIDTH && 
           coord.second >= 0 && coord.second < NOC_HEIGHT;
}

void CongestionModel::update_link_utilization(int node_id, DIR direction, bool active) {
    if (node_id < 0 || node_id >= NODE_NUM || direction > LOCAL) return;
    
    // Exponential moving average for utilization
    const double alpha = 0.1; // Smoothing factor
    double new_value = active ? 1.0 : 0.0;
    link_utilization[node_id][direction] = 
        alpha * new_value + (1 - alpha) * link_utilization[node_id][direction];
        
    // Update congestion info
    congestion_matrix[node_id][direction].utilization = link_utilization[node_id][direction];
    congestion_matrix[node_id][direction].last_update = sc_time_stamp();
}

double CongestionModel::get_link_congestion(int node_id, DIR direction) {
    if (node_id < 0 || node_id >= NODE_NUM || direction > LOCAL) return 1.0;
    return link_utilization[node_id][direction];
}

CongestionInfo CongestionModel::get_node_congestion(int node_id) {
    if (node_id < 0 || node_id >= NODE_NUM) return CongestionInfo();
    
    CongestionInfo info;
    double total_util = 0.0;
    int active_ports = 0;
    
    for (int dir = 0; dir <= LOCAL; ++dir) {
        total_util += link_utilization[node_id][dir];
        if (link_utilization[node_id][dir] > 0.01) active_ports++;
    }
    
    info.utilization = active_ports > 0 ? total_util / active_ports : 0.0;
    info.last_update = sc_time_stamp();
    return info;
}

void CongestionModel::update_congestion() {
    // Update congestion statistics every clock cycle
    for (int i = 0; i < NODE_NUM; ++i) {
        for (int dir = 0; dir <= LOCAL; ++dir) {
            // Decay utilization over time if not updated
            sc_time time_diff = sc_time_stamp() - congestion_matrix[i][dir].last_update;
            if (time_diff > sc_time(10, SC_NS)) {
                link_utilization[i][dir] *= 0.95; // Decay factor
            }
        }
    }
}

bool CongestionModel::is_link_congested(int node_id, DIR direction) {
    double congestion = get_link_congestion(node_id, direction);
    return congestion > CONGESTION_THRESHOLD_HIGH;
}

std::vector<DIR> CongestionModel::get_available_directions(int node_id, std::pair<int,int> dst_coord) {
    std::vector<DIR> available;
    std::pair<int,int> curr_coord = id2coord(node_id);
    
    // Check each direction for feasibility and congestion
    if (dst_coord.first > curr_coord.first && curr_coord.first < NOC_WIDTH - 1) {
        if (!is_link_congested(node_id, E)) available.push_back(E);
    }
    if (dst_coord.first < curr_coord.first && curr_coord.first > 0) {
        if (!is_link_congested(node_id, W)) available.push_back(W);
    }
    if (dst_coord.second > curr_coord.second && curr_coord.second < NOC_HEIGHT - 1) {
        if (!is_link_congested(node_id, S)) available.push_back(S);
    }
    if (dst_coord.second < curr_coord.second && curr_coord.second > 0) {
        if (!is_link_congested(node_id, N)) available.push_back(N);
    }
    
    return available;
}

} // namespace LLM_SIM_NOC