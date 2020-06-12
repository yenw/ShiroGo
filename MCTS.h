//
// Created by yuanyu on 2017.12.28.
//

#pragma once

#include <vector>
#include <unordered_map>
#include "MCTSMsg.h"
#include "MCTSParam.h"
#include "Utils.h"
#include "UCTNode.h"
#include "GoBoard.h"
#include "Network.h"

class MCTS
{
public:
    MCTS(GoBoard board);
    ~MCTS();
    int GenMove(Network *nn, const MCTSParam& param);
    bool Play(int p, bool show = true);
    void CollectData(MCTSGame& mcts_game, int play_p, bool is_greedy);
    bool GameOver(){ return m_board.game_over(); }
    int FinalScore(){ return m_board.score_tt(); }

    // EarlyStop & EarlyResign
    bool EarlyStop();
    int EarlyStopScore();
    int EarlyResign();

    // VK
    void InitVirtualKomi(Network* nn, const MCTSParam& param);

    // MC & UCB & UCT
    int MCGenMove(Network* nn, uint64_t simulations, bool is_train = true);
    int UCBGenMove(Network* nn, uint64_t simulations, bool is_train = true);
    int UCTGenMove(Network* nn, const MCTSParam& param);

private:
    // MCTS
    void DirichletNoise(UCTNode* node, float epsilon, float alpha);
    void Expand(UCTNode* node, GoBoard& board);
    UCTNode* Advantage(UCTNode* node);
    UCTNode* Select(UCTNode* node);
    void AddNoise(bool is_train);
    void Search(UCTNode* node, GoBoard& board, MCTSResult& result);
    UCTNode* MaxLCB(UCTNode* node);
    UCTNode* MaxVisit(UCTNode* node);
    UCTNode* RandomVisit(UCTNode* node);
    UCTNode* BestChild(UCTNode* node);
    double EarlyStop(const vector<double>& board_value);
    void PV(GoBoard& board, vector<float>& policy, vector<float>& value);
    void ShowPV();

    // MCTS Data
    void CollectData(MCTSData& data, UCTNode* node, GoBoard& board, int play_p, bool is_greedy, float lr);
    void CollectNode(MCTSGame& mcts_game, UCTNode* node, int skip_p, int depth, vector<int>& path);

    // UCT
    void UCTSearch(UCTNode* node, GoBoard& board, MCTSResult& result);

private:
    uint64_t m_total_visits;
    uint64_t m_total_adv_visits;
    uint64_t m_total_nn_visits;
    uint64_t m_total_data_count;
    uint64_t m_total_swap_count;
    uint64_t m_genmove_visits;
    uint64_t m_genmove_adv_visits;
    uint64_t m_genmove_nn_visits;
    uint64_t m_genmove_data_count;
    uint64_t m_genmove_swap_count;
    GoBoard m_board;
    UCTNode* m_root;
    float m_virtual_komi;
    MCTSParam m_param;
    Network* m_nn;
};
