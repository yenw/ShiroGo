//
// Created by yuanyu on 2017.12.28.
//

#include <iostream>
#include <limits>
#include <math.h>
#include <random>
#include "MCTS.h"
#include "Random.h"
#include "ReplayPool.h"
#include "Transform.h"

MCTS::MCTS(GoBoard board)
    :m_board{board}
{
    m_total_visits = 0;
    m_total_nn_visits = 0;
    m_total_adv_visits = 0;
    m_total_data_count = 0;
    m_total_swap_count = 0;
    m_virtual_komi = 0.0f;
    m_root = new UCTNode();
}

MCTS::~MCTS()
{
    UCTNode::Free(m_root);
    m_root = nullptr;
}

void MCTS::DirichletNoise(UCTNode* node, float epsilon, float alpha)
{
    auto dirichlet_vector = std::vector<float>{};
    std::gamma_distribution<float> gamma(alpha, 1.0f);

    for (size_t i = 0; i < node->child_count; ++i)
        dirichlet_vector.emplace_back(gamma(GoRandom::Get()));

    auto sample_sum = std::accumulate(begin(dirichlet_vector), end(dirichlet_vector), 0.0f);
    if ( sample_sum < std::numeric_limits<float>::min() )
        return;

    for (auto& v: dirichlet_vector)
        v /= sample_sum;

    auto child = node->child;
    for (auto noise: dirichlet_vector)
    {
        child->policy = (1.0f - epsilon) * child->nn_policy + epsilon * noise;
        child = child->next;
    }
}

UCTNode* MCTS::Select(UCTNode* node)
{
    auto size = m_board.get_width() * m_board.get_height();
    auto n = node->n;
    auto child = (UCTNode*)nullptr;

    auto best_child = (UCTNode*)nullptr;
    auto best_ucb = std::numeric_limits<double>::lowest();
    auto child_ucb{0.0};

    auto max_child = (UCTNode*)nullptr;
    auto max_q = std::numeric_limits<double>::lowest();
    auto child_q{0.0};

    auto best_nodes = int{0};
    auto fpu{0.0};
    switch (m_param.fpu_mode)
    {
        case FPUMode::Parent_NN:
            fpu = -node->nn_value;
            break;
        case FPUMode::Parent_MCTS:
            fpu = -node->value;
            break;
    }

    // 动态更新策略
    if (m_param.use_pi_sync)
    {
        uint64_t n_sync = m_param.pi_sync * m_param.simulations;
        if (node->n % n_sync == 0)
        {
            auto n_sum = uint64_t{0};
            child = m_root->child;
            while (child)
            {
                n_sum += child->n;
                child = child->next;
            }

            if (n_sum != 0)
            {
                child = m_root->child;
                while (child)
                {
                    child->policy = static_cast<decltype(child->policy)>(child->n) / n_sum;
                    child = child->next;
                }
            }
        }
    }

    // uct
    auto node_var = std::max(node->variance, 1.0);
    auto node_is = sqrt(node->value * node->value + node_var);
    auto child_var{0.0};
    auto bound{0.0};

    child = node->child;
    while (child)
    {
        if (child->evaluated)
            child_q = child->value;
        else
            child_q = fpu;

        switch (m_param.puct_mode)
        {
            case PUCTMode::NCK_STDEV:
                child_var = std::max(child->variance, 1.0);
                bound = size * m_param.puct_size_k;
                bound += sqrt(node_var) * m_param.puct_node_k + sqrt(child_var) * m_param.puct_child_k;
                child_ucb = child_q + m_param.c_puct * bound * child->policy * sqrt(n) / (child->n + 1);
                break;
            case PUCTMode::NODE_IS:
                bound = node_is;
                child_ucb = child_q + m_param.c_puct * bound * child->policy * sqrt(n) / (child->n + 1);
                break;
        }

        if (child_ucb > best_ucb)
        {
            best_ucb = child_ucb;
            best_child = child;
            best_nodes = 1;
        }
        else if (child_ucb == best_ucb)
        {
            ++best_nodes;
        }

        if (child_q > max_q)
        {
            max_q = child_q;
            max_child = child;
        }

        child = child->next;
    }

    // 使用网络时，下面可以注释掉
    if (best_nodes > 1)
    {
        auto best_select = GoRandom::Get().FastR31(best_nodes);
        child = node->child;
        while (child)
        {
            if (child->evaluated)
                child_q = child->value;
            else
                child_q = fpu;

            switch (m_param.puct_mode)
            {
                case PUCTMode::NCK_STDEV:
                    // Paper: size_k = 0.5, node_k = 0.5, child_k = 0.0
                    child_var = std::max(child->variance, 1.0);
                    bound = size * m_param.puct_size_k;
                    bound += sqrt(node_var) * m_param.puct_node_k + sqrt(child_var) * m_param.puct_child_k;
                    child_ucb = child_q + m_param.c_puct * bound * child->policy * sqrt(n) / (child->n + 1);
                    break;
                case PUCTMode::NODE_IS:
                    // Importance Sampling for Online Planning under Uncertainty
                    bound = node_is;
                    child_ucb = child_q + m_param.c_puct * bound * child->policy * sqrt(n) / (child->n + 1);
                    break;
            }

            if (child_ucb == best_ucb)
            {
                if (best_select == 0)
                {
                    best_child = child;
                    break;
                }
                else
                    --best_select;
            }
            child = child->next;
        }
    }

    // explore
    best_child->explore = (best_child != max_child);
    return best_child;
}

void MCTS::Expand(UCTNode* node, GoBoard& board)
{
    auto last_child = (UCTNode*)nullptr;
    auto child_count = int{0};
    auto policy = vector<float>();
    auto board_value = vector<float>();
    PV(board, policy, board_value);

    // policy
    auto policy_max = 0.0f;
    auto policy_sum = 0.0f;
    for (auto p: board.empty_point())
    {
        if ( board.is_legal(p) )
        {
            auto new_child = new UCTNode();
            new_child->p = p;
            if (last_child)
            {
                last_child->next = new_child;
                last_child = last_child->next;
            }
            else
            {
                last_child = new_child;
                node->child = last_child;
            }
            ++child_count;

            int np = board.pos_no_pad(p);
            policy_sum += policy[np];
            policy_max = max(policy_max, policy[np]);
        }
    }

    // PASS 统计
    if ( board.legal_pass() )
    {
        ++child_count;
        policy_sum += policy.back();
    }

    // 默认策略, 防止网络输出全0
    if (policy_sum == 0.0f)
    {
        policy_sum = float(child_count);
        for (auto& p: policy)
            p = 1.0f;
    }

    // value
    auto value = std::accumulate(begin(board_value), end(board_value), 0.0);// 黑视角分数
    value += board.score_bonus(node->nn_board_value, m_param.bonus);// 黑视角死子奖励

    // 父节点分数
    node->child_count = child_count;
    node->nn_board_value = vector<double>(begin(board_value), end(board_value));

    if ( board.next_black() )// 下一手黑棋, 说明子节点是黑节点
        value = -value;// 因此node节点是白节点, 需要负值.

    node->nn_value = value;

    // 子节点策略
    if (node->child)
    {
        // 处理棋盘上的点
        auto child = node->child;
        while (child)
        {
            int np = board.pos_no_pad(child->p);
            child->policy = policy[np] / policy_sum;// 由网络给出真实值
            child->nn_policy = float(child->policy);
            child = child->next;
        }

        // 处理PASS点
        if ( board.legal_pass() )
        {
            auto pass_child = new UCTNode();
            pass_child->p = GoBoard::PASS_MOVE;
            pass_child->policy = policy.back() / policy_sum;
            pass_child->nn_policy = float(pass_child->policy);

            if (last_child)
                last_child->next = pass_child;
            else
                node->child = pass_child;
        }
    }
    else// 对方无棋可下. 只能pass
    {
        auto pass_child = new UCTNode();
        pass_child->p = GoBoard::PASS_MOVE;
        pass_child->nn_policy = 1.0f;
        pass_child->policy = 1.0f;
        if (last_child)
            last_child->next = pass_child;
        else
            node->child = pass_child;
    }
}

UCTNode* MCTS::Advantage(UCTNode* node)
{
    auto child = node->child;
    if ( !child || !m_param.use_advantage )
        return nullptr;

    if ( !child->next )
        return child;

    auto policy_1st = 0.0f;
    auto policy_2nd = 0.0f;
    auto child_1st = (UCTNode*)nullptr;
    //auto child_2nd = (UCTNode*)nullptr;
    while (child)
    {
        auto policy = child->policy;
        if (policy > policy_2nd)
        {
            if (policy > policy_1st)
            {
                policy_2nd = policy_1st;
                policy_1st = policy;
                //child_2nd = child_1st;
                child_1st = child;
            }
            else
            {
                policy_2nd = policy;
                //child_2nd = child;
            }
        }
        child = child->next;
    }

    if (policy_1st > policy_2nd + m_param.advantage_thres)
    {
        ++m_genmove_adv_visits;
        return child_1st;
    }
    else
        return nullptr;
}

void MCTS::Search(UCTNode* node, GoBoard& board, MCTSResult& result)
{
    auto update_dynamic_result = [](UCTNode* node, float value, float virtual_komi, bool next_black)
    {
        auto win_result = 0.0;
        auto loss_result = 0.0;
        if ( !next_black )
        {
            // 下一步白棋, 当前是黑节点
            if ( value > virtual_komi + 4.0 )
                win_result = 1.0;// 黑棋获胜
            else if ( value < virtual_komi - 4.0 )
                loss_result = 1.0;// 黑棋落败
        }
        else
        {
            // 下一步黑棋, 当前是白节点
            if ( -value < virtual_komi - 4.0 )
                win_result = 1.0;// 白棋获胜
            else if ( -value > virtual_komi + 4.0 )
                loss_result = 1.0;// 白棋落败
        }

        // 特定颜色方的胜率/负率. 胜率越高, 节点越好.
        node->dynamic_win_rate += (win_result - node->dynamic_win_rate) / node->n;
        node->dynamic_loss_rate += (loss_result - node->dynamic_loss_rate) / node->n;
    };

    if ( board.game_over() )
    {
        // 双pass/最大步长 对局结束
        result.board_value = board.score_bv();
        result.early_stop = 1.0;
        result.value = std::accumulate(begin(result.board_value), end(result.board_value), 0.0f);// 黑视角分数
        result.value += board.score_bonus(result.board_value, m_param.bonus);// 黑视角死子奖励

        // 转换视角
        if ( board.next_black() )
            result.value = -result.value;

        ++node->n;
        ++node->n_exp;
        node->nn_value = result.value;
        node->nn_board_value = result.board_value;
        node->early_stop = result.early_stop;

        node->value = node->nn_value;
        node->board_value = node->nn_board_value;

        node->variance = 0.0;
        node->depth_avg = board.move_count();
        node->evaluated = true;

        // dynamic_result
        update_dynamic_result(node, result.value, m_virtual_komi, board.next_black());
        return;
    }

    auto select_node = (UCTNode*)nullptr;
    if ( !node->child )
    {
        // 扩展子节点
        Expand(node, board);
        select_node = Advantage(node);
        if ( !select_node )
        {
            // 无明显占优子节点, 本次探索结束.
            result.board_value = node->nn_board_value;
            result.value = node->nn_value;
            result.early_stop = EarlyStop(node->nn_board_value);

            ++node->n;
            ++node->n_exp;
            node->value = node->nn_value;
            node->board_value = node->nn_board_value;
            node->early_stop = result.early_stop;

            node->variance = 0.0;
            node->depth_avg = board.move_count();
            node->evaluated = true;

            // dynamic_result
            update_dynamic_result(node, result.value, m_virtual_komi, board.next_black());
            return;
        }

        // else 继续探索.
    }
    else
        select_node = Select(node);

    // before search
    auto select_n_exp = select_node->n_exp;
    auto p = select_node->p;
    auto next_black = board.next_black();
    board.move_at(p);//board.board_show();
    Search(select_node, board, result);

    // after search: result
    result.value = -result.value;

    // after search: node
    ++node->n;
    ++node->n_exp;
    node->child_n_exp += select_node->n_exp - select_n_exp;

    if ( !node->evaluated )
    {
        node->value = result.value;
        node->board_value = result.board_value;
        node->early_stop = result.early_stop;
        node->variance = 0.0;
        node->depth_avg = board.move_count();
        node->evaluated = true;
    }
    else
    {
        // explore && old - new > thres
        if (node->explore && (node->value - result.value > m_param.exp_thres))
            --node->n_exp;

        // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        double n = node->n;
        auto new_value = node->value + (result.value - node->value) / n;// new = old + (new - old) / n
        node->variance = ((n - 1) * node->variance + (result.value - node->value) * (result.value - new_value)) / n;
        node->value = new_value;
        node->depth_avg = node->depth_avg + (board.move_count() - node->depth_avg) / n;
        node->early_stop = node->early_stop + (result.early_stop - node->early_stop) / n;
        for (size_t i = 0; i < result.board_value.size(); ++i)
            node->board_value[i] = node->board_value[i] + (result.board_value[i] - node->board_value[i]) / n;// new = old + (new - old) / n
    }
    node->explore = false;

    // dynamic_result
    update_dynamic_result(node, result.value, m_virtual_komi, next_black);
}

UCTNode* MCTS::MaxLCB(UCTNode* node)
{
    auto best_node = (UCTNode*)nullptr;
    auto n = uint64_t{0};
    auto best_lcb = -1000000.0f;
    auto child = node->child;
    while (child)
    {
        if (m_param.use_exp_policy)
            n = child->n_exp;
        else
            n = child->n;

        if (n > 0)
        {
            auto child_lcb = child->value - m_param.c_puct * sqrt(child->variance / child->n);
            if (child_lcb > best_lcb)
            {
                best_lcb = child_lcb;
                best_node = child;
            }
        }

        child = child->next;
    }
    return best_node;
}

UCTNode* MCTS::MaxVisit(UCTNode* node)
{
    auto best_node = (UCTNode*)nullptr;
    auto n = uint64_t{0};
    auto best_n = uint64_t{0};
    auto best_value = -1000000.0f;
    auto child = node->child;
    while (child)
    {
        if (m_param.use_exp_policy)
            n = child->n_exp;
        else
            n = child->n;

        if (n > best_n)
        {
            best_n = n;
            best_value = child->value;
            best_node = child;
        }
        else if (n == best_n)
        {
            if (child->value > best_value)
            {
                best_n = n;
                best_value = child->value;
                best_node = child;
            }
        }
        child = child->next;
    }
    return best_node;
}

UCTNode* MCTS::RandomVisit(UCTNode* node)
{
    auto size = m_board.get_width() * m_board.get_height();
    auto best_node = (UCTNode*)nullptr;
    auto n = 0.0;
    auto n_sum = 0.0;
    auto n_pruned = m_param.n_pruned;
    auto n_child = vector<double>();
    auto n_cdf = vector<double>();
    auto child_vec = vector<UCTNode*>();
    auto child = node->child;
    while (child)
    {
        if (m_param.use_exp_policy)
            n = child->n_exp;
        else
            n = child->n;

        if (n > n_pruned)
        {
            n_sum += n;
            n_child.push_back(n);
            child_vec.push_back(child);
        }

        child = child->next;
    }

    // 分布至[0, 1]
    for (auto& n_i: n_child)
        n_i /= n_sum;

    // CDF
    n_sum = 0.0;
    for (auto n_i: n_child)
    {
        n_sum += n_i;
        n_cdf.push_back(n_sum);
    }

    // 跟据比例, 选择子节点
    if ( !n_cdf.empty() )
    {
        double r = GoRandom::Get().RFloat();
        for (size_t i = 0; i < n_child.size(); ++i)
        {
            if (n_cdf[i] >= r)
            {
                best_node = child_vec[i];
                break;
            }
        }
    }
    return best_node;
}

UCTNode* MCTS::BestChild(UCTNode* node)
{
    // 选择子节点
    auto best_node = (UCTNode*)nullptr;
    if (m_param.is_greedy)
    {
        if (m_param.play_mode == PlayMode::LCB)
            best_node = MaxLCB(node); // 选择子节点中具有最大LCB的节点.
        else
            best_node = MaxVisit(node); // 常规MCTS, 选择访问次数最多的节点
    }
    else
    {
        best_node = RandomVisit(node); // 按照概率选择节点
    }

    return best_node;
}

void MCTS::AddNoise(bool is_train)
{
    auto noise_alpha = m_param.dir_alpha / (m_board.get_width() * m_board.get_height());
    auto noise_epsilon = m_param.dir_epsilon;

    if ( !m_root->child )
    {
        GoBoard board = m_board;
        Expand(m_root, board);

        // Noise
        if (is_train)
            DirichletNoise(m_root, noise_epsilon, noise_alpha);

        auto select_node = Advantage(m_root);
        if (select_node)
        {
            // 继续探索
            MCTSResult result;

            // play & search
            auto select_n_exp = select_node->n_exp;
            auto p = select_node->p;
            board.move_at(p);
            Search(select_node, board, result);
            result.value = -result.value;

            ++m_root->n;
            ++m_root->n_exp;
            m_root->child_n_exp += select_node->n_exp - select_n_exp;

            m_root->value = result.value;
            m_root->board_value = result.board_value;

            m_root->variance = 0.0;
            m_root->depth_avg = board.move_count();
            m_root->evaluated = true;
        }
        else
        {
            ++m_root->n;
            ++m_root->n_exp;

            m_root->value = m_root->nn_value;
            m_root->board_value = m_root->nn_board_value;

            m_root->variance = 0.0;
            m_root->depth_avg = board.move_count();
            m_root->evaluated = true;
        }
    }
    else
    {
        if (m_root->child->next)
        {
            // Noise
            if (is_train)
                DirichletNoise(m_root, noise_epsilon, noise_alpha);
        }
    }
}

int MCTS::GenMove(Network* nn, const MCTSParam& param)
{
    // init
    m_param = param;
    m_nn = nn;
    m_genmove_visits = 0;
    m_genmove_nn_visits = 0;
    m_genmove_adv_visits = 0;
    m_genmove_data_count = 0;
    m_genmove_swap_count = 0;

    // 根节点添加噪音
    AddNoise(m_param.is_train);

    // 只此一手
    if (m_root->child && !m_root->child->next)
        return m_root->child->p;

    auto StopSim = [&]()
    {
        switch (m_param.sim_mode)
        {
            case SimulationMode::Playout:
                return false;
            case SimulationMode::Visit:
                return m_root->n >= m_param.simulations;
            case SimulationMode::Visit_Exp:
                return m_root->child_n_exp >= m_param.simulations;
        }
        return false;
    };

    // 进行搜索
    MCTSResult result;
    for (uint64_t i = 0; i < m_param.simulations; ++i)
    {
        if (StopSim())
            break;

        GoBoard board = m_board;
        Search(m_root, board, result);
        ++m_total_visits;
        ++m_genmove_visits;
    }
    m_total_nn_visits += m_genmove_nn_visits;
    m_total_adv_visits += m_genmove_adv_visits;

    // 选择子节点
    auto best_node = BestChild(m_root);
    if (best_node)
        return best_node->p;

    if ( m_board.max_move() )
        std::cerr << "MAX Move" << std::endl;

    return GoBoard::NO_MOVE;
}

void MCTS::InitVirtualKomi(Network* nn, const MCTSParam& param)
{
    if ( !param.is_train )
        return;

    // 1 搜索, 获取落子值
    GenMove(nn, param);

    // 2 确定虚拟贴目
    m_virtual_komi = -m_root->value;

    // 3 恢复原状
    // 3.1 清空统计量
    m_total_visits = 0;
    m_total_nn_visits = 0;
    m_total_adv_visits = 0;

    // 3.2 清空树
    UCTNode::Free(m_root);
    m_root = nullptr;
    m_root = new UCTNode();
}

bool MCTS::Play(int p, bool show)
{
    if (show)
        ShowPV();

    auto size = m_board.get_width() * m_board.get_height();
    auto GetSR = [size](UCTNode* node)
    {
        auto score_max = double(-size);
        auto score_min = double(size);
        auto child = node->child;
        while (child)
        {
            if (child->n > 1)
            {
                score_max = max(score_max, child->value);
                score_min = min(score_min, child->value);
            }
            child = child->next;
        }
        auto score_range = abs(score_max - score_min) * 0.5;
        return score_range;
    };

    auto new_root = m_root->child;
    auto prev_node = (UCTNode*)nullptr;
    while (new_root)
    {
        if (new_root->p == p)
        {
            if (prev_node == nullptr)
                m_root->child = new_root->next;
            else
                prev_node->next = new_root->next;

            new_root->next = nullptr;
            m_board.move_at(p);

            if (show)
            {
                // Info
                m_board.board_show();

                auto move_count = m_board.move_count();
                auto next_black = m_board.next_black();
                auto sign = next_black * 2 - 1;
                auto s_sr = GetSR(m_root);
                auto sa_sr = GetSR(new_root);
                auto s_sd = sqrt(m_root->variance);
                auto sa_sd = sqrt(new_root->variance);
                auto s_c = (s_sr + s_sd) / 2.0;
                auto sa_c = (sa_sr + sa_sd) / 2.0;

                auto po_avg = m_total_visits / move_count;
                auto nn_avg = m_total_nn_visits / move_count;
                auto adv_avg = m_total_adv_visits / move_count;
                auto data_avg = m_total_data_count / float(move_count);
                auto swap_avg = m_total_swap_count / float(move_count);

                fprintf(stderr, "MAX(%d), EXP(%d), ADV(%d), SYN(%d), ", m_param.is_greedy, m_param.use_exp_policy, m_param.use_advantage, m_param.use_pi_sync);
                fprintf(stderr, "P(%d), SIM(%d), SP(%d), FPU(%d), ", m_param.play_mode, m_param.sim_mode, m_param.sp_mode, m_param.fpu_mode);
                fprintf(stderr, "ES(%d), ER(%d), RD(%d)\n", m_param.use_early_stop, m_param.use_early_resign, USE_RD);
                fprintf(stderr, "VT Komi: %-6.3f PO: %-6" PRIu64 "\n", m_virtual_komi, m_param.simulations);
                fprintf(stderr, "PO_GM : %-6" PRIu64 " NN_GM : %-6" PRIu64 , m_genmove_visits, m_genmove_nn_visits);
                fprintf(stderr, " ADV_GM : %-6" PRIu64 " DT_GM : %-6" PRIu64 " SW_GM : %-6" PRIu64 "\n", m_genmove_adv_visits, m_genmove_data_count, m_genmove_swap_count);
                fprintf(stderr, "PO_AVG: %-6" PRIu64 " NN_AVG: %-6" PRIu64 , po_avg, nn_avg);
                fprintf(stderr, " ADV_AVG: %-6" PRIu64 " DT_AVG: %-6.3f SW_AVG: %-6.3f\n", adv_avg, data_avg, swap_avg);
                fprintf(stderr, "N_V: %-6" PRIu64 " EN_V: %-6" PRIu64 " SR_V: %-7.3f STD_V: %-7.3f C_V: %-7.3f", m_root->n, m_root->child_n_exp, s_sr, s_sd, s_c);
                fprintf(stderr, " D_V: %-5.1f ES_V: %-3.1f MCTS_V: %-7.3f", m_root->depth_avg - move_count + 1, m_root->early_stop, m_root->value * sign);
                if (next_black)
                    fprintf(stderr, " WR_V: %-5.3f LR_V: %-5.3f\n", m_root->dynamic_win_rate, m_root->dynamic_loss_rate);
                else
                    fprintf(stderr, " WR_V: %-5.3f LR_V: %-5.3f\n", m_root->dynamic_loss_rate, m_root->dynamic_win_rate);
                fprintf(stderr, "N_Q: %-6" PRIu64 " EN_Q: %-6" PRIu64 " SR_Q: %-7.3f STD_Q: %-7.3f C_Q: %-7.3f", new_root->n, new_root->child_n_exp, sa_sr, sa_sd, sa_c);
                fprintf(stderr, " D_Q: %-5.1f ES_Q: %-3.1f MCTS_Q: %-7.3f", new_root->depth_avg - move_count + 1, new_root->early_stop, -new_root->value * sign);
                if (next_black)
                    fprintf(stderr, " WR_Q: %-5.3f LR_Q: %-5.3f", new_root->dynamic_loss_rate, new_root->dynamic_win_rate);
                else
                    fprintf(stderr, " WR_Q: %-5.3f LR_Q: %-5.3f", new_root->dynamic_win_rate, new_root->dynamic_loss_rate);
                std::cerr << std::endl;
            }

            UCTNode::Free(m_root);
            m_root = new_root;

            if ( m_param.is_train && !m_param.use_subtree )
            {
                UCTNode::Free(m_root);
                m_root = new UCTNode();
            }

            return true;
        }
        prev_node = new_root;
        new_root = new_root->next;
    }

    m_board.move_at(p);
    return false;
}

void MCTS::PV(GoBoard& board, vector<float>& policy, vector<float>& board_value)
{
    // 随机旋转
    int mode = 1;
    static int mode4[] = {1, 4, 5, 8};
    static int mode8[] = {1, 2, 3, 4, 5, 6, 7, 8};
    static int recover8[] = {1, 2, 7, 4, 5, 6, 3, 8};

    auto width = board.get_width();
    auto height = board.get_height();
    if (width == height)
    {
        auto mode_idx = GoRandom::Get().RangeR31(0, 8);
        mode = mode8[mode_idx];
    }
    else
    {
        auto mode_idx = GoRandom::Get().RangeR31(0, 4);
        mode = mode4[mode_idx];
    }

    // 黑白互换
    int shuffle = GoRandom::Get().RangeR31(0, 2);

    vector<float> planes_rotate, policy_rotate, board_value_rotate;
    board.get_planes01(planes_rotate, INPUT_HISTORY, mode, shuffle, GoBoard::NO_MOVE, ACT_TYPE == "selu");
    m_nn->predict(planes_rotate, policy_rotate, board_value_rotate);

    // 求出旋转后策略和值
    if (mode == 1)
    {
        policy = policy_rotate;
        board_value = board_value_rotate;
    }
    else
    {
        auto recover_mode = recover8[mode - 1];
        policy.resize(policy_rotate.size());
        board_value.resize(board_value_rotate.size());
        transform_policy(policy.data(), policy_rotate.data(), width, height, recover_mode);
        transform_board_value(board_value.data(), board_value_rotate.data(), width, height, recover_mode, 0);
    }

    if (shuffle)
    {
        for (auto& v: board_value)
            v = -v;
    }
    ++m_genmove_nn_visits;
}

void MCTS::CollectData(MCTSData& data, UCTNode* node, GoBoard& board, int play_p, bool is_greedy, float lr)
{
    auto width = board.get_width();
    auto height = board.get_height();
    auto board_size = width * height;
    board.get_planes01(data.stone, INPUT_HISTORY, 1, 0, play_p, 0);

    vector<float> q_value(board_size + 1, 0.0f);
    vector<float> policy(board_size + 1, 0.0f);
    vector<float> exp_policy(board_size + 1, 0.0f);
    vector<float> board_value(board_size, 0.0f);
    auto n_pruned = m_param.n_pruned;
    auto child = node->child;
    auto visit = float{0};
    auto visit_exp = float{0};
    while (child)
    {
        if (child->n > n_pruned)
            visit += child->n;

        if (child->n_exp > n_pruned)
            visit_exp += child->n_exp;

        child = child->next;
    }

    // policy & q_value & loss
    child = node->child;
    while (child)
    {
        auto p = child->p;
        auto np = p;
        if (p == GoBoard::PASS_MOVE)
            np = board_size;
        else
            np = board.pos_no_pad(p);

        if (child->n_exp > n_pruned)
            exp_policy[np] = float(child->n_exp) / visit_exp;

        if (child->n > n_pruned)
            policy[np] = float(child->n) / visit;

        q_value[np] = float(child->value);
        child = child->next;
    }

    // board value & loss
    for (size_t i = 0; i < node->board_value.size(); ++i)
        board_value[i] = float(node->board_value[i]);

    // value
    auto value = std::accumulate(begin(board_value), end(board_value), 0.0f);

    // play
    data.is_greedy = is_greedy;
    if (play_p == GoBoard::PASS_MOVE)
        data.np = board_size;
    else
        data.np = board.pos_no_pad(play_p);

    if (m_param.use_exp_policy)
        data.policy.swap(exp_policy);
    else
        data.policy.swap(policy);

    // state
    if (ACTION_PLANES > 0)
    {
        GoBoard board_tmp = board;
        board_tmp.play(play_p);
        board_tmp.get_state(data.state);
    }

    data.board_value.swap(board_value);
    data.q_value.swap(q_value);
    data.value = value;
    data.next_black = board.next_black();
    data.swap_action = board.swap_action() && USE_SWAP_ACTION;
    data.lr = lr;
}

void MCTS::CollectNode(MCTSGame& mcts_game, UCTNode* node, int skip_p, int depth, vector<int>& path)
{
    auto get_lr = [&]()
    {
        switch (m_param.sim_mode)
        {
            case SimulationMode::Playout:
            case SimulationMode::Visit:
                return float(node->n) / m_param.simulations;
                break;
            case SimulationMode::Visit_Exp:
                return float(node->child_n_exp) / m_param.simulations;
                break;
        }
        return 0.0f;
    };

    auto skip_n = [&]()
    {
        auto lr = get_lr();
        return lr < m_param.rd_rate;
    };

    if (!node)
        return;

    // Visit Next
    if (depth == 1)
        CollectNode(mcts_game, node->next, skip_p, depth, path);
    else
        CollectNode(mcts_game, node->next, GoBoard::NO_MOVE, depth, path);

    // Skip the subtree and self
    if (depth == 1 && node->p == skip_p)
        return;

    // Visit Child
    if (depth == 0)
        CollectNode(mcts_game, node->child, skip_p, depth + 1, path);
    else
    {
        path.push_back(node->p);
        CollectNode(mcts_game, node->child, GoBoard::NO_MOVE, depth + 1, path);
        path.pop_back();
    }

    // Visit Self
    if (!node->child)
        return;

    if (skip_n())
        return;

    if (depth > 0)
    {
        MCTSData data;

        // play path
        GoBoard board = m_board;
        for (auto p: path)
            board.play(p);

        // play self
        board.play(node->p);
        
        // lr
        auto lr = min(1.0f, get_lr());

        // play_p
        int rand_p = GoBoard::NO_MOVE;
        auto rand = GoRandom::Get().FastR31(node->child_count);
        auto child = node->child;
        while (child)
        {
            if (rand == 0)
                rand_p = child->p;

            --rand;
            child = child->next;
        }

        CollectData(data, node, board, rand_p, false, lr);
        mcts_game.sample.emplace_back(std::move(data));
        ++m_genmove_data_count;
    }
}

void MCTS::CollectData(MCTSGame& mcts_game, int play_p, bool is_greedy)
{
    MCTSData data;

    // root data
    CollectData(data, m_root, m_board, play_p, is_greedy, 1.0f);
    mcts_game.game.emplace_back(std::move(data));
    m_genmove_swap_count += m_board.swap_action();
    m_total_swap_count += m_genmove_swap_count;

    // tree data
    if (is_greedy || !USE_RD)
        return;

    int best_p = GoBoard::NO_MOVE;
    auto best_n = uint64_t{0};
    auto n = best_n;
    auto child = m_root->child;
    while (child)
    {
        if (m_param.use_exp_policy)
            n = child->n_exp;
        else
            n = child->n;

        if (n > best_n)
        {
            best_n = n;
            best_p = child->p;
        }
        child = child->next;
    }

    // skip the best child;
    if (best_p != play_p)
        return;

    vector<int> path;
    CollectNode(mcts_game, m_root, play_p, 0, path);
    m_total_data_count += m_genmove_data_count;
}

int MCTS::EarlyResign()
{
    // 返回值: 1 黑赢; -1 白赢; 0 在和棋区间内
    if (m_root == nullptr)
        return 0;

    if ( m_board.move_count() * 4 < m_board.get_width() * m_board.get_height() )
        return 0;

    if ( m_board.next_black() )// 下一步黑, 当前为白节点
    {
        if (m_root->dynamic_loss_rate > 0.95)// 白输, 即黑赢
            return 1;
        else if (m_root->dynamic_win_rate > 0.95)// 白赢, 即黑输
            return -1;
    }
    else// 下一步白, 当前为黑节点
    {
        if (m_root->dynamic_win_rate > 0.95)// 黑赢
            return 1;
        else if (m_root->dynamic_loss_rate > 0.95)// 白赢
            return -1;
    }

    return 0;
}

bool MCTS::EarlyStop()
{
    if (m_root == nullptr)
        return false;

    if ( m_board.move_count() * 2 < m_board.get_width() * m_board.get_height() )
        return false;

    for (auto& v: m_root->board_value)
    {
        if (v < 0.9 && v > -0.9)
            return false;
    }

    return true;
}

double MCTS::EarlyStop(const vector<double>& board_value)
{
    auto early_stop = 0.0;
    for (auto& v: board_value)
    {
        if (v > 0.0)
            early_stop += min(v / 0.9, 1.0);
        else
            early_stop += min(-v / 0.9, 1.0);
    }

    return early_stop / board_value.size();
}

int MCTS::EarlyStopScore()
{
    if (m_root == nullptr)
        return 0;

    int score = 0;
    for (auto v: m_root->board_value)
    {
        if (v > 0.9)
            ++score;
        else if (v < -0.9)
            --score;
    }

    return score;
}

void MCTS::ShowPV()
{
    // 获取数据
    auto width = m_board.get_width();
    auto height = m_board.get_height();
    auto board_size = width * height;

    vector<float> c_puct(board_size + 1, 0.0f);
    vector<float> cp(board_size + 1, 0.0f);
    vector<float> policy(board_size + 1, 0.0f);
    vector<float> nn_policy(board_size + 1, 0.0f);
    vector<float> exp_policy(board_size + 1, 0.0f);
    vector<float> q_value(board_size + 1, 0.0f);
    vector<float> q_value_nn(board_size + 1, 0.0f);
    vector<float> lcb_value(board_size + 1, 0.0f);
    vector<bool> lcb_value_bool(board_size + 1, false);
    auto board_value = m_root->board_value;
    auto nn_board_value = m_root->nn_board_value;

    auto n_pruned = m_param.n_pruned;
    auto child = m_root->child;
    auto visit = float{0};
    auto visit_exp = float{0};
    while (child)
    {
        if (child->n > n_pruned)
            visit += child->n;

        if (child->n_exp > n_pruned)
            visit_exp += child->n_exp;

        child = child->next;
    }

    child = m_root->child;
    while (child)
    {
        auto p = child->p;
        auto np = p;
        if (p == GoBoard::PASS_MOVE)
            np = board_size;
        else
            np = m_board.pos_no_pad(p);

        if (child->n > n_pruned)
            policy[np] = float(child->n) / visit;

        if (child->n_exp > n_pruned)
            exp_policy[np] = float(child->n_exp) / visit_exp;

        q_value[np] = child->value;
        q_value_nn[np] = child->nn_value;
        nn_policy[np] = child->nn_policy;
        if (child->n > 1)
        {
            lcb_value[np] = q_value[np] - m_param.c_puct * sqrt(child->variance / child->n);
            lcb_value_bool[np] = true;
        }

        auto node_var = std::max(m_root->variance, 1.0);
        auto child_var = std::max(child->variance, 1.0);
        c_puct[np] = board_size * m_param.puct_size_k;
        c_puct[np] += sqrt(node_var) * m_param.puct_node_k + sqrt(child_var) * m_param.puct_child_k;
        cp[np] = c_puct[np] * policy[np];

        child = child->next;
    }

    auto p_max = [](vector<float>& policy)
    {
        auto policy_idx = size_t{0};
        auto policy_max = 0.0f;
        for (size_t i = 0; i < policy.size(); ++i)
        {
            if (policy[i] > policy_max)
            {
                policy_max = policy[i];
                policy_idx = i;
            }
        }
        return policy_idx;
    };

    auto policy_max_idx = p_max(policy);
    auto nn_policy_max_idx = p_max(nn_policy);
    auto exp_policy_max_idx = p_max(exp_policy);

    auto q_max = [&policy](vector<float>& q)
    {
        auto q_idx = size_t{0};
        auto q_max = -10000000.0f;
        for (size_t i = 0; i < q.size(); ++i)
        {
            if (q[i] > q_max && policy[i] != 0.0f)
            {
                q_max = q[i];
                q_idx = i;
            }
        }
        return q_idx;
    };

    auto q_value_nn_max_idx = q_max(q_value_nn);
    auto q_value_max_idx = q_max(q_value);
    auto c_puct_max_idx = q_max(c_puct);
    auto cp_max_idx = q_max(cp);
    auto lcb_value_max_idx = size_t{0};
    {
        auto lcb_value_max = -10000000.0f;
        for (size_t i = 0; i < lcb_value.size(); ++i)
        {
            if (lcb_value[i] > lcb_value_max && lcb_value_bool[i])
            {
                lcb_value_max = lcb_value[i];
                lcb_value_max_idx = i;
            }
        }
    }

    // 控制台颜色
    auto B = "\033[1;30m";
    auto W = "\033[1;37m";
    auto C = "\033[1;35m";
    auto G = "\033[0;32m";
    auto R = "\033[0;31m";
    auto None = "\033[0m";

    // lambda
    auto output_p_bv = [&](const string& title, size_t policy_idx, vector<float>& policy_o, vector<double>& bv_o)
    {
        cerr << title << endl;
        cerr << "      ";
        for (int x = 0; x < width; ++x)
            cerr << G << (x) % 10 << None << "   ";

        cerr << "        ";
        for (int x = 0; x < width; ++x)
            cerr << G << (x) % 10 << None << "     ";

        cerr << endl;

        for (int y = 0; y < height; ++y)
        {
            // policy
            if (y > 9)
                cerr << " ";
            else
                cerr << "  ";
            cerr << G << y << None;
            for (int x = 0; x < width; ++x)
            {
                if (x + y * width == policy_idx)
                {
                    cerr << R;
                    fprintf(stderr, "%4d", min(999, int(policy_o[x + y * width] * 1000)));
                    cerr << None;
                }
                else
                    fprintf(stderr, "%4d", min(999, int(policy_o[x + y * width] * 1000)));
            }

            if (y > 9)
                cerr << " " << G << y << None;
            else
                cerr << " " << G << y << " " << None;

            // board_value
            if (y > 9)
                cerr << " ";
            else
                cerr << "  ";

            cerr << G << y << None;
            for (int x = 0; x < width; ++x)
                fprintf(stderr, "%6.2f", bv_o[x + y * width] * 0.998);

            if (y > 9)
                cerr << " " << G << y << None;
            else
                cerr << " " << G << y << " " << None;
            cerr << endl;
        }

        cerr << "      ";
        for (int x = 0; x < width; ++x)
            cerr << G << (x) % 10 << None << "   ";

        cerr << "        ";
        for (int x = 0; x < width; ++x)
            cerr << G << (x) % 10 << None << "     ";

        cerr << endl;
        auto space_policy = (width + 1) / 2 * 4 + 3;
        auto space_value = width * 4 + 6 - space_policy + (width + 1) / 2 * 6 + 3;
        if (board_size == policy_idx)
        {
            cerr << R;
            fprintf(stderr, "%*d", space_policy, min(999, int(policy_o[board_size] * 1000)));
            cerr << None;
        }
        else
            fprintf(stderr, "%*d", space_policy, min(999, int(policy_o[board_size] * 1000)));

        auto v = std::accumulate(begin(bv_o), end(bv_o), 0.0f);
        fprintf(stderr, "%*.1f", space_value, v);
        cerr << endl;
    };

    auto output_q = [&](string title, size_t idx, vector<float>& q, bool auto_sign = true)
    {
        int sign;
        if (auto_sign)
            sign = m_board.next_black() * 2 - 1;
        else
            sign = 1;
        cerr << title << endl;
        cerr << "        ";
        for (int x = 0; x < width; ++x)
            cerr << G << (x) % 10 << None << "     ";

        cerr << endl;
        for (int y = 0; y < height; ++y)
        {
            // q
            if (y > 9)
                cerr << " ";
            else
                cerr << "  ";
            cerr << G << y << None;
            for (int x = 0; x < width; ++x)
            {
                if (x + y * width == idx)
                {
                    cerr << R;
                    fprintf(stderr, "%6.1f", sign * q[x + y * width]);
                    cerr << None;
                }
                else
                    fprintf(stderr, "%6.1f", sign * q[x + y * width]);
            }

            if (y > 9)
                cerr << " " << G << y << None;
            else
                cerr << " " << G << y << " " << None;

            cerr << endl;
        }

        cerr << "        ";
        for (int x = 0; x < width; ++x)
            cerr << G << (x) % 10 << None << "     ";

        cerr << endl;
        auto space_q = (width + 1) / 2 * 6 + 3;
        if (board_size == idx)
        {
            cerr << R;
            fprintf(stderr, "%*.1f", space_q, sign * q[board_size]);
            cerr << None;
        }
        else
            fprintf(stderr, "%*.1f", space_q, sign * q[board_size]);

        cerr << endl;
    };

    auto output_p = [&](string title, size_t idx, vector<float>& policy_o)
    {
        cerr << title << endl;
        cerr << "      ";
        for (int x = 0; x < width; ++x)
            cerr << G << (x) % 10 << None << "   ";

        cerr << endl;

        for (int y = 0; y < height; ++y)
        {
            // policy
            if (y > 9)
                cerr << " ";
            else
                cerr << "  ";
            cerr << G << y << None;
            for (int x = 0; x < width; ++x)
            {
                if (x + y * width == idx)
                {
                    cerr << R;
                    fprintf(stderr, "%4d", min(999, int(policy_o[x + y * width] * 1000)));
                    cerr << None;
                }
                else
                    fprintf(stderr, "%4d", min(999, int(policy_o[x + y * width] * 1000)));
            }

            if (y > 9)
                cerr << " " << G << y << None;
            else
                cerr << " " << G << y << " " << None;

            cerr << endl;
        }

        cerr << "      ";
        for (int x = 0; x < width; ++x)
            cerr << G << (x) % 10 << None << "   ";

        cerr << endl;
        auto space_policy = (width + 1) / 2 * 4 + 3;

        if (board_size == idx)
        {
            cerr << R;
            fprintf(stderr, "%*d", space_policy, min(999, int(policy_o[board_size] * 1000)));
            cerr << None;
        }
        else
            fprintf(stderr, "%*d", space_policy, min(999, int(policy_o[board_size] * 1000)));

        cerr << endl;
    };

    // 输出c值
    output_q("c_puct", c_puct_max_idx, c_puct, false);
    output_q("cp", cp_max_idx, cp, false);

    // 输出动作值
    if (m_nn)
        output_q("q_value_nn", q_value_nn_max_idx, q_value_nn);

    output_q("q_value", q_value_max_idx, q_value);
    output_q("lcb_value", lcb_value_max_idx, lcb_value);

    // 输出NN
    if (m_nn)
        output_p_bv("nn_policy/nn_board_value", nn_policy_max_idx, nn_policy, nn_board_value);

    // 输出MCTS
    output_p_bv("policy(" + std::to_string(int(visit)) + ")/board_value", policy_max_idx, policy, board_value);

    // 输出Label值
    if (m_nn)
        output_p("exp_policy(" + std::to_string(int(visit_exp)) + ")", exp_policy_max_idx, exp_policy);
}

int MCTS::MCGenMove(Network* nn, uint64_t simulations, bool is_train)
{
    m_nn = nn;

    // 只此一手
    if (m_root->child && !m_root->child->next)
        return m_root->child->p;

    // 寻找合理节点
    auto child_list = vector<int>();
    auto child_value = vector<double>();
    auto child = m_root->child;
    auto child_count = 0;
    for (auto p: m_board.empty_point())
    {
        if ( m_board.is_legal(p) )
        {
            auto new_child = new UCTNode();
            new_child->p = p;
            if (child)
            {
                child->next = new_child;
                child = child->next;
            }
            else
            {
                child = new_child;
                m_root->child = child;
            }
            child_list.push_back(p);
            ++child_count;
        }
    }

    // PASS 节点
    if ( m_board.legal_pass() )
    {
        auto pass_child = new UCTNode();
        pass_child->p = GoBoard::PASS_MOVE;
        child_list.push_back(GoBoard::PASS_MOVE);
        ++child_count;

        if (child)
            child->next = pass_child;
        else
            m_root->child = pass_child;
    }

    // 设置初值
    child = m_root->child;
    while (child)
    {
        child->nn_policy = 1.0f / child_count;
        child->nn_board_value = vector<double>(m_board.get_width() * m_board.get_height(), 0);
        child->value = 0;
        child->policy = child->nn_policy;
        child->board_value = child->nn_board_value;
        child = child->next;
    }

    if ( !m_root->evaluated )
    {
        m_root->evaluated = true;
        m_root->n = 0;
        m_root->value = 0;
        m_root->nn_board_value = vector<double>(m_board.get_width() * m_board.get_height(), 0);
        m_root->board_value = m_root->nn_board_value;
    }

    // 进行搜索
    auto time_start = std::chrono::steady_clock::now();
    auto steps = uint64_t{0};
    auto value = 0.0f;
    for (uint64_t i = 1; i <= simulations; ++i)
    {
        GoBoard board = m_board;
        vector<double> board_value;

        // MC落子
        auto idx = GoRandom::Get().FastR31(child_count);
        board.move_at(child_list[idx]);
        ++steps;

        // MC随机落子
        while ( !board.game_over() )
        {
            auto ep_legal = vector<int>();
            for (auto ep: board.empty_point())
            {
                if ( board.is_legal(ep) )
                    ep_legal.push_back(ep);

                if ( board.legal_pass() )
                    ep_legal.push_back(GoBoard::PASS_MOVE);
            }

            auto rnd_idx = GoRandom::Get().FastR31(ep_legal.size());
            board.move_at(ep_legal[rnd_idx]);
            ++steps;
        }

        // MC统计
        board_value = board.score_bv();
        value = std::accumulate(begin(board_value), end(board_value), 0.0f);// 黑视角分数
        if ( !m_board.next_black() )
            value = -value;

        // 更新节点统计
        child = m_root->child;
        ++m_root->n;
        auto p = child_list[idx];
        while (child)
        {
            if (child->p == p)
            {
                child->evaluated = true;
                ++child->n;
                child->value = child->value + (value - child->value) / i;
                for (size_t k = 0; k < board_value.size(); ++k)
                    child->board_value[k] = child->board_value[k] + (board_value[k] - child->board_value[k]) / i;// new = old + (new - old) / n

                m_root->value = m_root->value + (-value - m_root->value) / i;
                for (size_t k = 0; k < board_value.size(); ++k)
                    m_root->board_value[k] = m_root->board_value[k] + (board_value[k] - m_root->board_value[k]) / i;// new = old + (new - old) / n

                break;
            }
            child = child->next;
        }
    }
    auto time_end = std::chrono::steady_clock::now();
    auto sec = std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_start).count();
    cerr << "po/s: " << simulations / sec << endl;
    cerr << "move/s: " << steps / sec << endl;

    // 选择最优子节点
    auto best_node = (UCTNode*)nullptr;
    auto best_value = -1000000.0;
    child = m_root->child;
    while (child)
    {
        if (child->value > best_value)
        {
            best_node = child;
            best_value = child->value;
        }
        child = child->next;
    }

    return best_node->p;
}

int MCTS::UCBGenMove(Network* nn, uint64_t simulations, bool is_train)
{
    auto size = m_board.get_width() * m_board.get_height();
    m_nn = nn;

    // 只此一手
    if (m_root->child && !m_root->child->next)
        return m_root->child->p;

    // 寻找合理节点
    auto child_list = vector<int>();
    auto child_value = vector<double>();
    auto child = m_root->child;
    auto child_count = 0;
    for (auto p: m_board.empty_point())
    {
        if ( m_board.is_legal(p) )
        {
            auto new_child = new UCTNode();
            new_child->p = p;
            if (child)
            {
                child->next = new_child;
                child = child->next;
            }
            else
            {
                child = new_child;
                m_root->child = child;
            }
            child_list.push_back(p);
            ++child_count;
        }
    }

    // PASS 节点
    if ( m_board.legal_pass() )
    {
        auto pass_child = new UCTNode();
        pass_child->p = GoBoard::PASS_MOVE;
        child_list.push_back(GoBoard::PASS_MOVE);
        ++child_count;

        if (child)
            child->next = pass_child;
        else
            m_root->child = pass_child;
    }

    // 设置初值
    child = m_root->child;
    while (child)
    {
        child->nn_policy = 1.0f / child_count;
        child->nn_board_value = vector<double>(m_board.get_width() * m_board.get_height(), 0);
        child->value = 0;
        child->policy = child->nn_policy;
        child->board_value = child->nn_board_value;
        child = child->next;
    }

    if ( !m_root->evaluated )
    {
        m_root->evaluated = true;
        m_root->n = 0;
        m_root->value = 0;
        m_root->nn_board_value = vector<double>(m_board.get_width() * m_board.get_height(), 0);
        m_root->board_value = m_root->nn_board_value;
    }

    // 进行搜索
    auto time_start = std::chrono::steady_clock::now();
    auto steps = uint64_t{0};
    auto value = 0.0f;
    for (uint64_t i = 1; i <= simulations; ++i)
    {
        GoBoard board = m_board;
        vector<double> board_value;
        ++m_root->n;

        // UCB选择
        auto select_child = (UCTNode*)nullptr;
        auto ucb_max = std::numeric_limits<double>::lowest();
        auto ucb_value = 0.0;

        child = m_root->child;
        while (child)
        {
            ucb_value = child->value + size * m_param.c_puct * sqrt(log(m_root->n) / (child->n + 1));
            if (ucb_value > ucb_max)
            {
                select_child = child;
                ucb_max = ucb_value;
            }

            child = child->next;
        }


        board.move_at(select_child->p);
        ++steps;

        // MC随机落子
        while ( !board.game_over() )
        {
            auto p = int(GoBoard::PASS_MOVE);
            auto ep = board.empty_point();

            if ( !ep.empty() )
            {
                bool legal_pass = board.legal_pass();
                int max_idx = int(ep.size());
                while (max_idx > 0)
                {
                    auto rnd_idx = GoRandom::Get().RangeR31(0, max_idx + legal_pass);
                    if (rnd_idx == max_idx)
                    {
                        p = GoBoard::PASS_MOVE;
                        break;
                    }

                    if ( board.is_legal(ep[rnd_idx]) )
                    {
                        p = ep[rnd_idx];
                        break;
                    }
                    else
                    {
                        --max_idx;
                        ep[rnd_idx] = ep[max_idx];
                    }
                }
            }

            board.move_at(p);
            ++steps;
        }

        // MC统计
        board_value = board.score_bv();
        value = std::accumulate(begin(board_value), end(board_value), 0.0f);// 黑视角分数
        if ( !m_board.next_black() )
            value = -value;

        // 更新节点统计
        select_child->evaluated = true;
        ++select_child->n;
        select_child->value = select_child->value + (value - select_child->value) / i;
        for (size_t k = 0; k < board_value.size(); ++k)
            select_child->board_value[k] = select_child->board_value[k] + (board_value[k] - select_child->board_value[k]) / i;// new = old + (new - old) / n

        m_root->value = m_root->value + (-value - m_root->value) / i;
        for (size_t k = 0; k < board_value.size(); ++k)
            m_root->board_value[k] = m_root->board_value[k] + (board_value[k] - m_root->board_value[k]) / i;// new = old + (new - old) / n
    }

    auto time_end = std::chrono::steady_clock::now();
    auto sec = std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_start).count();
    cerr << "po/s: " << simulations / sec << endl;
    cerr << "step/s: " << steps / sec << endl;

    // 选择最优子节点
    auto best_node = (UCTNode*)nullptr;
    auto best_value = -1000000.0;
    child = m_root->child;
    while (child)
    {
        if (child->value > best_value)
        {
            best_node = child;
            best_value = child->value;
        }
        child = child->next;
    }

    return best_node->p;
}

void MCTS::UCTSearch(UCTNode* node, GoBoard& board, MCTSResult& result)
{
    if ( board.game_over() )
    {
        // 双pass/最大步长 对局结束
        result.board_value = board.score_bv();
        result.value = std::accumulate(begin(result.board_value), end(result.board_value), 0.0f);// 黑视角分数
        if ( board.next_black() )
            result.value = -result.value;

        ++node->n;
        ++node->n_exp;
        node->nn_value = result.value;
        node->nn_board_value = result.board_value;

        node->value = node->nn_value;
        node->board_value = node->nn_board_value;

        node->variance = 0.0;
        node->depth_avg = board.move_count();
        node->evaluated = true;
        return;
    }

    auto select_node = (UCTNode*)nullptr;
    if ( !node->child )
    {
        // 扩展子节点
        {
            // 寻找合理节点
            auto child = node->child;
            auto child_count = 0;
            for (auto p: board.empty_point())
            {
                if ( board.is_legal(p) )
                {
                    auto new_child = new UCTNode();
                    new_child->p = p;
                    if (child)
                    {
                        child->next = new_child;
                        child = child->next;
                    }
                    else
                    {
                        child = new_child;
                        node->child = child;
                    }
                    ++child_count;
                }
            }

            // PASS 节点
            if ( board.legal_pass() )
            {
                auto pass_child = new UCTNode();
                pass_child->p = GoBoard::PASS_MOVE;
                ++child_count;

                if (child)
                    child->next = pass_child;
                else
                    node->child = pass_child;
            }

            // 设置初值
            child = node->child;
            while (child)
            {
                child->nn_policy = 1.0f / child_count;
                child->nn_board_value = vector<double>(board.get_width() * board.get_height(), 0);
                child->value = 0;
                child->policy = child->nn_policy;
                child->board_value = child->nn_board_value;
                child = child->next;
            }

            // 记录有多少个动作
            node->child_count = child_count;
        }

        // 随机rollout至终局
        {
            // 提前记录
            bool next_black = board.next_black();

            // 随机rollout至终局
            while ( !board.game_over() )
            {
                auto p = int(GoBoard::PASS_MOVE);
                auto ep = board.empty_point();

                if ( !ep.empty() )
                {
                    bool legal_pass = board.legal_pass();
                    int max_idx = int(ep.size());
                    while (max_idx > 0)
                    {
                        auto rnd_idx = GoRandom::Get().RangeR31(0, max_idx + legal_pass);
                        if (rnd_idx == max_idx)
                        {
                            p = GoBoard::PASS_MOVE;
                            break;
                        }

                        if ( board.is_legal(ep[rnd_idx]) )
                        {
                            p = ep[rnd_idx];
                            break;
                        }
                        else
                        {
                            --max_idx;
                            ep[rnd_idx] = ep[max_idx];
                        }
                    }
                }

                board.move_at(p);
            }

            // 统计
            result.board_value = board.score_bv();
            result.value = std::accumulate(begin(result.board_value), end(result.board_value), 0.0f);// 黑视角分数
            if ( next_black )
                result.value = -result.value;
        }

        // 本次探索结束.
        ++node->n;
        ++node->n_exp;
        node->nn_value = result.value;
        node->nn_board_value = result.board_value;

        node->value = node->nn_value;
        node->board_value = node->nn_board_value;

        node->variance = 0.0;
        node->depth_avg = board.move_count();
        node->evaluated = true;
        return;
    }
    else
    {
        // UCB选择
        auto size = m_board.get_width() * m_board.get_height();
        auto n = node->n;

        auto best_child = (UCTNode*)nullptr;
        auto best_ucb = std::numeric_limits<double>::lowest();
        auto child_ucb = 0.0;

        auto max_child = (UCTNode*)nullptr;
        auto max_q = std::numeric_limits<double>::lowest();
        auto child_q{0.0};
        auto fpu = -node->value;

        // uct
        auto node_var = std::max(node->variance, 1.0);
        auto node_is = sqrt(node->value * node->value + node_var);
        auto child_var{0.0};
        auto bound{0.0};
        auto child = node->child;
        while (child)
        {
            if (child->evaluated)
                child_q = child->value;
            else
                child_q = fpu;

            switch (m_param.puct_mode)
            {
                case PUCTMode::NCK_STDEV:
                    // Paper: size_k = 0.5, node_k = 0.5, child_k = 0.0
                    child_var = std::max(child->variance, 1.0);
                    bound = size * m_param.puct_size_k;
                    bound += sqrt(node_var) * m_param.puct_node_k + sqrt(child_var) * m_param.puct_child_k;
                    child_ucb = child_q + m_param.c_puct * bound * child->policy * sqrt(n) / (child->n + 1);
                    break;
                case PUCTMode::NODE_IS:
                    // Importance Sampling for Online Planning under Uncertainty
                    bound = node_is;
                    child_ucb = child_q + m_param.c_puct * bound * child->policy * sqrt(n) / (child->n + 1);
                    break;
            }

            if (child_ucb > best_ucb)
            {
                best_child = child;
                best_ucb = child_ucb;
            }

            if (child_q > max_q)
            {
                max_q = child_q;
                max_child = child;
            }

            child = child->next;
        }
        best_child->explore = (best_child != max_child);
        select_node = best_child;
    }

    auto select_n_exp = select_node->n_exp;
    auto p = select_node->p;
    board.move_at(p);//board.board_show();
    UCTSearch(select_node, board, result);

    // after search: result
    result.value = -result.value;
    ++node->n;
    ++node->n_exp;
    node->child_n_exp += select_node->n_exp - select_n_exp;

    if ( !node->evaluated )
    {
        node->value = result.value;
        node->board_value = result.board_value;
        node->variance = 0.0;
        node->depth_avg = board.move_count();
        node->evaluated = true;
    }
    else
    {
        // explore
        if (node->explore && node->value > result.value)
            --node->n_exp;

        // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        double n = node->n;
        double new_value = node->value + (result.value - node->value) / n;// new = old + (new - old) / n
        node->variance = ((n - 1) * node->variance + (result.value - node->value) * (result.value - new_value)) / n;
        node->value = new_value;
        node->depth_avg = node->depth_avg + (board.move_count() - node->depth_avg) / n;
        for (size_t i = 0; i < result.board_value.size(); ++i)
            node->board_value[i] = node->board_value[i] + (result.board_value[i] - node->board_value[i]) / n;// new = old + (new - old) / n
    }
    node->explore = false;
}

int MCTS::UCTGenMove(Network *nn, const MCTSParam& param)
{
    auto size = m_board.get_width() * m_board.get_height();
    m_param = param;
    m_nn = nn;
    m_genmove_visits = 0;
    m_genmove_nn_visits = 0;
    m_genmove_adv_visits = 0;
    m_genmove_data_count = 0;
    m_genmove_swap_count = 0;

    // 只此一手
    if (m_root->child && !m_root->child->next)
        return m_root->child->p;

    if ( !m_root->evaluated )
    {
        m_root->n = 0;
        m_root->n_exp = 0;
        m_root->nn_value = 0;
        m_root->nn_board_value = vector<double>(m_board.get_width() * m_board.get_height(), 0);
        m_root->value = m_root->nn_value;
        m_root->board_value = m_root->nn_board_value;
        m_root->evaluated = true;
    }

    // 进行搜索
    auto prev_moves = m_root->depth_avg * m_root->n;
    auto time_start = std::chrono::steady_clock::now();
    MCTSResult result;
    for (uint64_t i = 1; i <= m_param.simulations; ++i)
    {
        GoBoard board = m_board;
        UCTSearch(m_root, board, result);
        ++m_total_visits;
        ++m_genmove_visits;
    }

    if (m_board.move_count() == 0)
        m_virtual_komi = -m_root->value;

    auto time_end = std::chrono::steady_clock::now();
    auto sec = std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_start).count();
    cerr << "po/s: " << int(m_param.simulations / sec) << endl;
    cerr << "move/s: " << int((m_root->depth_avg * m_root->n - prev_moves) / sec) << endl;

    // 选择最优子节点
    auto best_node = BestChild(m_root);
    if (best_node)
        return best_node->p;

    if (m_board.max_move() )
        cerr << "MAX Move" << std::endl;

    return GoBoard::NO_MOVE;
}
