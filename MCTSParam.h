//
// Created by yuanyu on 2020.05.17.
//

#ifndef SHIRO_MCTSPARAM_H
#define SHIRO_MCTSPARAM_H
#include <cstdint>

enum class SimulationMode
{
    Playout,
    Visit_Exp, // Visit <= Visit_Exp <= Playout
    Visit,
};

enum class PlayMode
{
    Visit, // N(s, a)
    LCB, // Lower Confident Bound
};

enum class SelfPlayMode
{
    MonteCarlo, // Play.Greedy + Train.MonteCarlo
    TreeBackup, // Play.Distribution + Train.TreeBackup
    Mixed,      // TB, TB, ..., MC, MC, MC
};

enum class FPUMode
{
    Parent_NN,   // -node->nn_value
    Parent_MCTS, // -node->value
};

// SPUCT,      // q + c * (size / 2 + sqrt(node.var) / 2) * child.policy * sqrt(n) / (child.n + 1);
// N2C_STDEV,  // q + c * (sqrt(node.var) * 2 + sqrt(child.var)) * child.policy * sqrt(n) / (child.n + 1);
// N2C2_STDEV, // q + c * (sqrt(node.var) * 2 + sqrt(child.var) * 2) * child.policy * sqrt(n) / (child.n + 1);
// NC2_STDEV,  // q + c * (sqrt(node.var) + sqrt(child.var) * 2) * child.policy * sqrt(n) / (child.n + 1);
// NC3_STDEV,  // q + c * (sqrt(node.var) + sqrt(child.var) * 3) * child.policy * sqrt(n) / (child.n + 1);
// NCK_STDEV,  // q + c * (sqrt(node.var) + sqrt(child.var) * K) * child.policy * sqrt(n) / (child.n + 1);
// SUM_STDEV,  // q + c * (sqrt(node.var) + sqrt(child.var)) * child.policy * sqrt(n) / (child.n + 1);
// AVG_STDEV,  // q + c * (sqrt(node.var) * 0.5 + sqrt(child.var) * 0.5) * child.policy * sqrt(n) / (child.n + 1);
// NODE_IS,    // q + c * sqrt(node.value^2 + node.var) * child.policy * sqrt(n) / (child.n + 1);

// NC1.5_STDEV
// C = 1.96 VS C = 1.5: 4X4 simulations=20000; 142W, 136D, 132L; avg_score: 0.195 
// C = 1.96 VS C = 2.5: 4X4 simulations=20000; 140W, 124D, 136L; avg_score: 0.235

// C = 1.96:
// NC1.5_STDEV VS NC1.25_STDEV: 6X4 simulations=20000;  368W,  31D, 361L; avg_score: 0.054 
// NC1.5_STDEV VS NC1.25_STDEV: 4X3 simulations=20000;  129W, 167D, 104L; avg_score: 0.27
// NC1.5_STDEV VS NC1.25_STDEV: 4X4 simulations=20000;  120W, 133D, 147L; avg_score: -0.54
// NC1.5_STDEV VS    NC2_STDEV: 4X4 simulations=20000;  144W, 137D, 119L; avg_score: 0.275

// NC2_STDEV VS  NC3_STDEV: 4X4 simulations=20000;  156W, 115D, 129L; avg_score: 0.71 
// NC2_STDEV VS  SUM_STDEV: 4X4 simulations=20000;  134W, 128D, 138L; avg_score: 0.06
// NC2_STDEV VS  AVG_STDEV: 4X4 simulations=20000;  201W,  85D, 114L; avg_score: 1.47
// NC2_STDEV VS    NODE_IS: 4X4 simulations=20000;  201W,  92D, 107L; avg_score: 1.54 
// NC2_STDEV VS      SPUCT: 4X4 simulations=20000;  148W, 115D, 137L; avg_score: 0.235

// NC3_STDEV VS  NC2_STDEV: 4X3 simulations=20000; 100W, 203D,  97L; avg_score: 0.295
// NC2_STDEV VS  SUM_STDEV: 4X3 simulations=20000; 131W, 162D, 107L; avg_score: 0.365
// NC2_STDEV VS N2C2_STDEV: 4X3 simulations=20000; 117W, 199D,  84L; avg_score: 0.075
// NC2_STDEV VS  N2C_STDEV: 4X3 simulations=20000; 118W, 175D, 107L; avg_score: 0.1175
// SUM_STDEV VS  AVG_STDEV: 4x3 simulations=20000; 142W, 165D,  93L; avg_score: 0.63
// SUM_STDEV VS    NODE_IS: 4x3 simulations=20000; 151W, 151D,  98L; avg_score: 0.37
// SUM_STDEV VS      SPUCT: 4x3 simulations=20000; 123W, 185D,  92L; avg_score: 0.3975
enum class PUCTMode
{
    NCK_STDEV, // q + c * (sqrt(node.var) + sqrt(child.var) * 1.5) * child.policy * sqrt(n) / (child.n + 1);
    NODE_IS, // q + c * sqrt(node.value^2 + node.var) * child.policy * sqrt(n) / (child.n + 1);
};

struct MCTSParam
{
    //float nn_lr = 0.01; PP: 0.128
    uint64_t simulations;
    uint64_t n_pruned = 1;// PP: 0; 落子 或 生成训练数据 时忽略小于等于n_pruned的子节点.
    PlayMode play_mode = PlayMode::Visit;// PP: Visit
    SimulationMode sim_mode = SimulationMode::Visit_Exp;// PP: Playout
    SelfPlayMode sp_mode = SelfPlayMode::Mixed;// Mixed方差大, 不一定稳定.
    PUCTMode puct_mode = PUCTMode::NCK_STDEV;
    FPUMode fpu_mode = FPUMode::Parent_MCTS;
    float bonus = 0.0f;// 0.0: 正常规则; 1.0 一子千金规则.
    float c_puct = 1.96f;// 越大探索性越强
    float puct_size_k = 0.0f;// PP: 0.5
    float puct_node_k = 1.0f;// PP: 0.5, 目前参数, 可用. 
    float puct_child_k = 1.5f;// PP: 0.0
    float advantage_thres = 0.6f;// Advantage Threshold
    float dir_alpha = 20.0f;// PP: 20.0; 越大, 分布越平均, 噪音越小
    float dir_epsilon = 0.25f;
    float pi_sync = 0.6f;// 每sim * pi_sync步, 用pi_label替换UCT公式中的pi;
    float mixed_min = 0.5f;// 在4x4上 min=0.25, max=0.75 貌似运行结果不好.
    float mixed_max = 1.5f;// min=0.5, max=1.5 在8x8上感觉基本不起太大作用.
    float rd_rate = 0.2f;
    float exp_thres = 0.1f;// node->value - result.value > exp_thres;
    bool use_subtree = true;
    bool use_advantage = true;
    bool use_early_stop = true;// PP.1: false; PP.2: true
    bool use_early_resign = true;// PP.1: false; PP.2: true
    bool use_exp_policy = true;// PP: false; 方差大, 不一定稳定.
    bool use_pi_sync = false;
    bool is_greedy = true;// 可变, 由SelfPlay根据SelfPlayMode进行改变.
    bool is_train = true;// 可变
};

#endif //SHIRO_MCTSPARAM_H