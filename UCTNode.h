//
// Created by yuanyu on 2017.12.28.
//

#pragma once

#include <cstdint>
#include <vector>

class UCTNode
{
public:
    UCTNode();
    UCTNode(float policy, float value, uint32_t p, uint64_t bv_size, uint64_t init_n = 0);
    static void Free(UCTNode* node);

    UCTNode* next;
    UCTNode* child;

    uint64_t n;
    uint64_t n_exp;
    uint64_t child_n_exp;
    double value;
    double nn_value;
    double variance;
    double dynamic_win_rate;
    double dynamic_loss_rate;
    double depth_avg;
    double early_stop;
    double policy;
    float nn_policy;

    std::vector<double> nn_board_value;
    std::vector<double> board_value;
    bool evaluated;
    bool explore;
    int p;
    int child_count;
};
