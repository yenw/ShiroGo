//
// Created by yuanyu on 2018.02.01.
//

#pragma once

#include <string>
#include "GoBoard.h"
#include "MCTS.h"
#include "Network.h"
#include "ReplayPool.h"
#include "Utils.h"

class SelfPlay
{
public:
    SelfPlay() = delete;
    SelfPlay(int width, int height, int simulations);
    void run(Network& nn, ReplayPool& pool, int games, bool is_train = true, bool warm_up = false);
    void mc_run(Network& nn, ReplayPool& pool, int games, bool is_train = true);
    void pk(Network& nn, int games, int komi);
    void profiler();

private:
    void ToPool(ReplayPool& pool);
    void SaveSGF(const string& fn, const string& sgf_bw);

private:
    int m_width, m_height;
    int m_simulations;
    int m_game_number;
    std::string sgf_head;
    std::string sgf_bw;
    StopWatch sw_genmove;
    MCTSGame mcts_game;
};
