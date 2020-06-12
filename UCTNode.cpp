//
// Created by yuanyu on 2017.12.28.
//

#include <limits>
#include "UCTNode.h"
#include "GoBoard.h"

UCTNode::UCTNode()
{
    next = nullptr;
    child = nullptr;
    n = 0;
    n_exp = 0;
    child_n_exp = 0;
    policy = 0.0;
    value = 0.0;
    nn_value = 0.0;
    variance = 0.0;
    dynamic_win_rate = 0.0;
    dynamic_loss_rate = 0.0;
    depth_avg = 0.0;
    early_stop = 0.0;
    nn_policy = policy;
    p = GoBoard::NO_MOVE;
    board_value.resize(0);
    nn_board_value.resize(0);
    evaluated = false;
    explore = false;
    child_count = 0;
}

UCTNode::UCTNode(float init_policy, float init_value, uint32_t position, uint64_t bv_size, uint64_t init_n)
{
    next = nullptr;
    child = nullptr;
    n = init_n;
    n_exp = init_n;
    child_n_exp = init_n;
    policy = init_policy;
    value = init_value;
    nn_value = init_value;
    variance = 0.0;
    dynamic_win_rate = 0.0;
    dynamic_loss_rate = 0.0;
    depth_avg = 0.0;
    early_stop = 0.0;
    nn_policy = init_policy;
    p = position;
    board_value.resize(bv_size, 0.0);
    nn_board_value.resize(bv_size, 0.0);
    evaluated = false;
    explore = false;
    child_count = 0;
}

void UCTNode::Free(UCTNode* node)
{
    if (node == nullptr)
        return;

    Free(node->child);
    Free(node->next);
    delete node;
}
