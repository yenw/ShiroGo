//
// Created by yuanyu on 2018.05.29.
//

#ifndef SHIRO_MCTSMSG_H
#define SHIRO_MCTSMSG_H
#include <vector>

struct MCTSData
{
    std::vector<float> stone;
    std::vector<float> policy;
    std::vector<float> board_value;
    std::vector<float> state;
    std::vector<float> q_value;
    float value;//sum(board_value);
    float lr;// 样本学习率
    int np;// 落子位置(无pad)
    bool is_greedy;// true: max, false: sample
    bool next_black;// 落子颜色
    bool swap_action;// 落子可交换
};

struct MCTSGame
{
    std::vector<MCTSData> game;// MCTS Game
    std::vector<MCTSData> sample;// MCTS Sample
};

struct MCTSResult
{
    std::vector<double> board_value;
    float value = 0.0f;
    float early_stop = 0.0f;
};

#endif //SHIRO_MCTSMSG_H
