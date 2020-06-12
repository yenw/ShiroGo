//
// Created by yuanyu on 2018.01.18.
//

#pragma once
#include <cstdint>
#include <vector>
#include <array>
#include "mxnet-cpp/MxNetCpp.h"
#include "MCTSMsg.h"
#include "Utils.h"
#include "config.h"
using namespace std;

class ReplayNode
{
public:
    void Init(int line_x, int line_y, int history_size);
    //void Swap(ReplayNode& node);

public:
    int m_line_x;
    int m_line_y;
    int m_planes;
    vector<int> m_action;// 前k步动作, ..., 前1步动作
    vector<float> m_stone;// stone + color + action
    vector<float> m_policy;
    vector<float> m_board_value;
    vector<float> m_state;
    float m_value;
    float m_lr;
};

class ReplayGame
{
public:
    void swap(ReplayGame& rg);

public:
    vector<ReplayNode> game;// 完整对局
    vector<ReplayNode> sample;// icing on the cake
};

class ReplayPool
{
public:
    ReplayPool(size_t size);
    void LoadData(int width, int height, int buffer_size);
    void SaveData(MCTSGame& mcts_game, int id, int final_score, int width, int height);
    void Put(ReplayGame& game);
    ReplayNode& Get(bool combined = false, bool uniform = false);
    void GetBatch(vector<mx_float>& stone_batch, vector<mx_float>& policy_batch, 
                  vector<mx_float>& board_value_batch, vector<mx_float>& state_batch, 
                  vector<mx_float>& lr_batch, uint32_t batch_size);
    size_t DataSize(){ return min(m_size, m_size - m_free); }

private:
    void LoadData(string data_folder, string rd_folder, int id);

private:
    StopWatch sw_1, sw_2, sw_3;
    vector<ReplayGame> m_pool;
    size_t m_size;
    size_t m_next;
    size_t m_free;

    float m_avg_length;
};
