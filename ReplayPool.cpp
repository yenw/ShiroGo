//
// Created by yuanyu on 2018.01.18.
//

#include <fstream>
#include "GoBoard.h"
#include "ReplayPool.h"
#include "Random.h"
#include "Transform.h"

void ReplayNode::Init(int line_x, int line_y, int history_size)
{
    m_line_x = line_x;
    m_line_y = line_y;
    m_planes = history_size * 2 + COLOR_PLANES + ACTION_PLANES;
    m_stone.clear();
    m_policy.clear();
    m_board_value.clear();
    m_state.clear();

    int board_size = line_x * line_y;
    m_stone.resize(m_planes * board_size, 0);
    m_policy.resize(board_size + 1, 0.0f);
    m_board_value.resize(board_size, 0.0f);
    m_state.resize(board_size * 3, 0.0f);
    m_value = 0.0f;
    m_lr = 1.0f;
}

void ReplayGame::swap(ReplayGame& rg)
{
    game.swap(rg.game);
    sample.swap(rg.sample);
}

ReplayPool::ReplayPool(size_t size)
{
    m_size = size;
    m_next = 0;
    m_free = size;
    m_avg_length = 0.0f;
    m_pool.resize(m_size);
}

void ReplayPool::Put(ReplayGame& game)
{
    if (m_free > 0)
        --m_free;

    m_pool[m_next].swap(game);
    m_next = (m_next + 1) % m_size;
}

ReplayNode& ReplayPool::Get(bool combined, bool uniform)
{
    // random game
    auto select_idx = int{0};
    if (uniform)
    {
        auto sample_size = size_t{0};
        for (auto& rg: m_pool)
            sample_size += (rg.game.size() + rg.sample.size());
        
        auto min_idx = size_t{0};
        auto max_idx = sample_size;
        auto rand_size = GoRandom::Get().RangeR31(min_idx, max_idx);
        
        sample_size = 0;
        for (int i = 0; i < m_pool.size(); ++i)
        {
            sample_size += (m_pool[i].game.size() + m_pool[i].sample.size());
            if (sample_size >= rand_size)
            {
                select_idx = i;
                break;
            }
        }
    }
    else
    {
        auto min_idx = size_t{0};
        auto max_idx = min(m_size, m_size - m_free);
        select_idx = GoRandom::Get().RangeR31(min_idx, max_idx);
    }
    
    if (combined)
    {
        auto new_idx = GoRandom::Get().FastR31(8 * SP_THREADS) + 1;
        select_idx = int32_t((m_next + m_size - new_idx) % m_size);
    }
    
    // random step
    auto& rg = m_pool[select_idx];
    auto step_min = 0;
    auto step_max = rg.game.size() + rg.sample.size();
    auto step_idx = GoRandom::Get().RangeR31(step_min, step_max);

    // norgal game or tree sample
    if (step_idx < rg.game.size())
        return rg.game[step_idx];
    else
        return rg.sample[step_idx - rg.game.size()];
}

void ReplayPool::LoadData(int width, int height, int buffer_size)
{
    auto max_id = get_file_id(width, height);
    if (max_id == -1)
    {
        cerr << "load data: error" << endl;
        exit(-1);
    }

    sw_3.clear_count();
    auto data_folder = get_data_folder(width, height);
    auto rd_folder = get_rd_folder(width, height);
    auto data_size = 0.0f;
    for (int i = max(0, max_id - buffer_size) + 1; i <= max_id; ++i)
    {
        LoadData(data_folder, rd_folder, i);
        ++data_size;
    }

    m_avg_length /= data_size;
    cerr << "avg length: " << m_avg_length << endl;
    sw_3.output_count("3");
}

void ReplayPool::LoadData(string data_folder, string rd_folder, int id)
{
    const char* p = nullptr;
    auto file_size = 0;
    auto curr = 0;
    auto is_eof = [&file_size, &curr]()
    {
        return curr == file_size;
    };

    auto read_str = [&p, &curr](size_t len) ->string
    {
        string str = string(p + curr, p + curr + len);
        curr += len;
        return str;
    };

    auto read_int = [&p, &curr]() ->int32_t
    {
        int32_t i;
        i = *(int32_t*)&p[curr];
        curr += 4;
        return i;
    };

    auto read_float = [&p, &curr]() ->float
    {
        float i;
        i = *(float*)&p[curr];
        curr += 4;
        return i;
    };

    ReplayGame rg;

    // normal game
    {
        auto fn_mcts = data_folder + "/" + to_string(id) + ".MCTS";
        sw_3.start_count();
        ifstream fin(fn_mcts.c_str(), ios::binary);
        string file((istreambuf_iterator<char>(fin)), istreambuf_iterator<char>());
        sw_3.end_count();

        p = file.c_str();
        file_size = file.size();
        curr = 0;

        if (read_str(4) != "MCTS")
        {
            cerr << "head: error" << endl;
            exit(-1);
        }

        auto version = read_int();
        auto handicap = read_int();
        auto height = read_int();
        auto width = read_int();
        auto final_score = read_int();
        auto board_size = height * width;

        if (version == 1)
        {
            vector<MCTSData> game_data;
            MCTSData data;
            while ( !is_eof() )
            {
                data.np = read_int();
                {
                    int flag = read_int();
                    data.next_black = flag & 0x1;
                    data.is_greedy = flag & 0x2;// version.1: false;
                    data.swap_action = flag & 0x4;// version.1: false;
                }
                data.value = read_float();

                // White & Black
                data.stone.resize(board_size * 2);
                for (int i = 0; i < board_size * 2; ++i)
                    data.stone[i] = read_float();

                data.board_value.resize(board_size);
                for (int i = 0; i < board_size; ++i)
                    data.board_value[i] = read_float();

                data.policy.resize(board_size + 1);
                for (int i = 0; i < board_size + 1; ++i)
                    data.policy[i] = read_float();

                data.q_value.resize(board_size + 1);
                for (int i = 0; i < board_size + 1; ++i)
                    data.q_value[i] = read_float();

                game_data.push_back(data);
            }
            m_avg_length += game_data.size();

            vector<float> state;
            GoBoard board(width, height);
            ReplayNode rn;
            for (int i = 0; i < game_data.size(); ++i)
            {
                auto& gd = game_data[i];
                rn.Init(width, height, INPUT_HISTORY);
                rn.m_value = gd.value;
                rn.m_policy = gd.policy;
                rn.m_board_value = gd.board_value;

                // stone
                // [W_{t-7}, B_{t-7}, ..., W_t, B_t, C_{white}, C{black}]
                int it = 0;
                for (int pl = 0; pl < INPUT_HISTORY; ++pl)
                {
                    auto h = max(0, i - (INPUT_HISTORY - pl) + 1);
                    auto& stone = game_data[h].stone;
                    copy(stone.begin(), stone.end(), rn.m_stone.begin() + it);
                    it += board_size * 2;
                }

                // color
                if (gd.next_black)
                    fill(rn.m_stone.begin() + it + board_size, rn.m_stone.begin() + it + board_size * 2, 1.0f);
                else
                    fill(rn.m_stone.begin() + it, rn.m_stone.begin() + it + board_size, 1.0f);

                it += board_size * 2;

                // action
                if (ACTION_PLANES > 0)
                {
                    if (gd.np != board_size)
                        rn.m_stone[it + gd.np] = 1.0f;

                    // state
                    if (gd.np == board_size)
                        board.play(GoBoard::PASS_MOVE);
                    else
                        board.play(board.pos(gd.np % width, gd.np / width));

                    board.get_state(state);
                    rn.m_state.swap(state);
                }

                if (gd.swap_action)
                    rn.m_action = GoBoard::calc_action(rn.m_stone, INPUT_HISTORY, board_size);

                // push
                rg.game.push_back(rn);
            }

            // mix value
            {
                auto smooth = [](vector<float>& mix_board_value)
                {
                    for (auto& v: mix_board_value)
                    {
                        if (v >= 0.9f)
                            v = 1.0f;
                        else if (v <= -0.9f)
                            v = -1.0f;
                    }
                };

                // for early stop
                auto& final_board_value = game_data.back().board_value;
                smooth(final_board_value);
                game_data.back().value = accumulate(final_board_value.begin(), final_board_value.end(), 0.0f);

                auto size = int(game_data.size());
                auto mix_board_value = game_data.back().board_value;
                auto mix_value = game_data.back().value;
                for (int i = size - 2; i >= 0; --i)
                {
                    const auto& gd = game_data[i];

                    // mix
                    if (!gd.is_greedy)
                    {
                        auto alpha = gd.policy[gd.np];
                        for (size_t k = 0; k < gd.board_value.size(); ++k)
                            mix_board_value[k] = alpha * mix_board_value[k] + (1.0 - alpha) * gd.board_value[k];

                        smooth(mix_board_value);
                        mix_value = accumulate(mix_board_value.begin(), mix_board_value.end(), 0.0f);
                    }

                    // update
                    rg.game[i].m_board_value = mix_board_value;
                    rg.game[i].m_value = mix_value;
                }
            }
        }
    }

    // sample
    if (USE_RD)
    {
        auto fn_sample = rd_folder + "/" + to_string(id) + ".RD";;// replay data
        sw_3.start_count();
        ifstream fin(fn_sample.c_str(), ios::binary);
        string file((istreambuf_iterator<char>(fin)), istreambuf_iterator<char>());
        sw_3.end_count();

        p = file.c_str();
        file_size = file.size();
        curr = 0;

        if (read_str(4) != "RPDT")
        {
            cerr << "head: error" << endl;
            exit(-1);
        }

        auto version = read_int();
        auto handicap = read_int();
        auto height = read_int();
        auto width = read_int();
        auto board_size = height * width;

        if (version == 1)
        {
            ReplayNode rn;
            MCTSData data;
            while ( !is_eof() )
            {
                rn.Init(width, height, INPUT_HISTORY);
                data.np = read_int();
                {
                    int flag = read_int();
                    data.next_black = flag & 0x1;
                    data.swap_action = flag & 0x4;// version.1: false;
                }
                rn.m_value = read_float();
                rn.m_lr = read_float();

                // (White & Black) * INPUT_HISTORY
                for (int i = 0; i < board_size * 2 * INPUT_HISTORY; ++i)
                    rn.m_stone[i] = read_float();

                for (int i = 0; i < board_size; ++i)
                    rn.m_board_value[i] = read_float();

                for (int i = 0; i < board_size + 1; ++i)
                    rn.m_policy[i] = read_float();

                data.q_value.resize(board_size + 1);
                for (int i = 0; i < board_size + 1; ++i)
                    data.q_value[i] = read_float();

                for (int i = 0; i < board_size * 3; ++i)
                    rn.m_state[i] = read_float();

                // color
                auto it = board_size * 2 * INPUT_HISTORY;
                if (data.next_black)
                    fill(rn.m_stone.begin() + it + board_size, rn.m_stone.begin() + it + board_size * 2, 1.0f);
                else
                    fill(rn.m_stone.begin() + it, rn.m_stone.begin() + it + board_size, 1.0f);

                it += board_size * 2;

                // action
                if (ACTION_PLANES > 0)
                {
                    if (data.np != board_size)
                        rn.m_stone[it + data.np] = 1.0f;
                }

                if (data.swap_action)
                    rn.m_action = GoBoard::calc_action(rn.m_stone, INPUT_HISTORY, board_size);

                rg.sample.push_back(rn);
            }
        }
    }

    Put(rg);
}

void ReplayPool::SaveData(MCTSGame& mcts_game, int id, int final_score, int width, int height)
{
    ofstream* data_file;
    auto write_str = [&data_file](const char* data, size_t len){ data_file->write(data, len); };
    auto write_int = [&data_file](int32_t data){ data_file->write((char*)&data, 4); };
    auto write_float = [&data_file](float data){ data_file->write((char*)&data, 4); };

    // normal game
    {
        auto fn_mcts = get_data_folder(width, height) + "/" + to_string(id) + ".MCTS";
        ofstream mcts_file(fn_mcts, ios::out|ios::binary);
        data_file = &mcts_file;

        write_str("MCTS", 4);
        write_int(1);// Version
        write_int(0);// Handicap
        write_int(height);// height
        write_int(width);// width
        write_int(final_score);// result
        for (const auto& data: mcts_game.game)
        {
            write_int(data.np);
            write_int(data.next_black | (int(data.is_greedy) << 1) | ((int(data.swap_action)) << 2));
            write_float(data.value);// sum(board_value), Black's view

            auto board_size = data.board_value.size();
            auto curr_board = board_size * (INPUT_HISTORY * 2 - 2);
            for (int i = 0; i < board_size * 2; ++i)
                write_float(data.stone[curr_board + i]);// White & Black

            for (const auto& board_value: data.board_value)
                write_float(board_value);// Black's view

            for (const auto& policy: data.policy)
                write_float(policy);// Player's view

            for (const auto& q_value: data.q_value)
                write_float(q_value);// Player's view
        }
        cerr << "Saving MCTS Data: " << fn_mcts << endl;
    }

    // sample
    if (USE_RD)
    {
        auto fn_sample = get_rd_folder(width, height) + "/" + to_string(id) + ".RD";;// replay data
        ofstream sample_file(fn_sample, ios::out|ios::binary);
        data_file = &sample_file;

        write_str("RPDT", 4);
        write_int(1);// Version
        write_int(0);// Handicap
        write_int(height);// height
        write_int(width);// width
        for (const auto& data: mcts_game.sample)
        {
            write_int(data.np);
            write_int(data.next_black | ((int(data.swap_action)) << 2));
            write_float(data.value);// sum(board_value), Black's view
            write_float(data.lr);

            auto board_size = data.board_value.size();
            for (int i = 0; i < board_size * 2 * INPUT_HISTORY; ++i)
                write_float(data.stone[i]);

            for (const auto& board_value: data.board_value)
                write_float(board_value);// Black's view

            for (const auto& policy: data.policy)
                write_float(policy);// Player's view

            for (const auto& q_value: data.q_value)
                write_float(q_value);// Player's view

            for (const auto& state: data.state)
                write_float(state);
        }
        cerr << "Saving Replay Data: " << fn_sample << endl;
    }
}

void ReplayPool::GetBatch(vector<mx_float>& stone_batch, vector<mx_float>& policy_batch, 
                          vector<mx_float>& board_value_batch, vector<mx_float>& state_batch, 
                          vector<mx_float>& lr_batch, uint32_t batch_size)
{
    int width, height;
    {
        auto node = Get();
        auto& stone = node.m_stone;
        auto& policy = node.m_policy;
        auto& board_value = node.m_board_value;
        auto& state = node.m_state;
        width = node.m_line_x;
        height = node.m_line_y;
        stone_batch.resize(batch_size * stone.size());
        policy_batch.resize(batch_size * policy.size());
        board_value_batch.resize(batch_size * board_value.size());
        state_batch.resize(batch_size * state.size());
        lr_batch.resize(batch_size);
    }

    auto transform_data = [&](int i, int mode, bool shuffle, bool no_action, bool swap_action, bool combined)
    {
        auto node = Get(combined);
        auto width = node.m_line_x;
        auto height = node.m_line_y;
        auto planes = node.m_planes;
        auto &stone = node.m_stone;
        auto &policy = node.m_policy;
        auto &board_value = node.m_board_value;
        auto &state = node.m_state;
        auto &action = node.m_action;
        auto lr = node.m_lr;
        auto action_idx = (width * height) * (INPUT_HISTORY * 2 + COLOR_PLANES);
        auto curr_board_idx = (width * height) * (INPUT_HISTORY * 2 - 2);
        
        if (ACTION_PLANES == 0)
        {
            // input
            transform_stone(&stone_batch[i * stone.size()], stone.data(), planes, width, height, mode, shuffle);

            // label
            transform_policy(&policy_batch[i * policy.size()], policy.data(), width, height, mode);
            transform_board_value(&board_value_batch[i * board_value.size()], board_value.data(), width, height, mode, shuffle);
        }
        else
        {
            // input
            transform_stone(&stone_batch[i * stone.size()], stone.data(), planes - ACTION_PLANES, width, height, mode, shuffle);
            transform_action(&stone_batch[i * stone.size() + action_idx], &stone[action_idx], ACTION_PLANES, width, height, mode, no_action);
            
            // label
            transform_policy(&policy_batch[i * policy.size()], policy.data(), width, height, mode);
            transform_board_value(&board_value_batch[i * board_value.size()], board_value.data(), width, height, mode, shuffle);
            transform_state(&state_batch[i * state.size()], state.data(), &stone[curr_board_idx], width, height, mode, shuffle, no_action);
        }

        if (swap_action && !action.empty())
        {
            auto action_new = GoBoard::shuffle_action(action);
            auto action_old = action;
            transform_stone_swap(&stone_batch[i * stone.size()], action_old.data(), action_new.data(), action_old.size(), INPUT_HISTORY * 2, width, height, mode, shuffle);
        }

        if (ACT_TYPE == "selu")
            transform_selu(&stone_batch[i * stone.size()], planes, width, height);

        // learning rate
        lr_batch[i] = lr;
    };

    bool is_shuffle = true;
    bool is_rotate = true;
    bool is_combined = true;

    vector<int> batch_mode(batch_size, 1);
    vector<bool> batch_shuffle(batch_size, false);
    vector<bool> batch_no_action(batch_size, false);
    vector<bool> batch_swap_action(batch_size, false);
    vector<bool> batch_combined(batch_size, false);

    if (is_rotate)
    {
        if (is_shuffle)
        {
            int div, rem;
            int normal_size = batch_size / 2;
            int shuffle_size = batch_size - normal_size;
            if (width == height)
            {
                // normal
                div = normal_size / 8;
                rem = normal_size - div * 8;
                for (int i = 0; i < rem; ++i)
                    batch_mode[i] = 1;

                for (int mode = 1; mode <= 8; ++mode)
                {
                    int start = rem + (mode - 1) * div;
                    int end = start + div;
                    for (int i = start; i < end && i < normal_size; ++i)
                        batch_mode[i] = mode;
                }

                // shuffle
                div = shuffle_size / 8;
                rem = shuffle_size - div * 8;
                for (int i = 0; i < rem; ++i)
                {
                    batch_mode[i + normal_size] = 1;
                    batch_shuffle[i + normal_size] = true;
                }

                for (int mode = 1; mode <= 8; ++mode)
                {
                    int start = rem + (mode - 1) * div;
                    int end = start + div;
                    for (int i = start; i < end && i < shuffle_size; ++i)
                    {
                        batch_mode[i + normal_size] = mode;
                        batch_shuffle[i + normal_size] = true;
                    }
                }
            }
            else
            {
                static int modes[] = {1, 4, 5, 8};

                // normal
                div = normal_size / 4;
                rem = normal_size - div * 4;
                for (int i = 0; i < rem; ++i)
                    batch_mode[i] = 1;

                for (int k = 0; k < 4; ++k)
                {
                    int mode = modes[k];
                    int start = rem + k * div;
                    int end = start + div;
                    for (int i = start; i < end && i < normal_size; ++i)
                        batch_mode[i] = mode;
                }

                // shuffle
                div = shuffle_size / 4;
                rem = shuffle_size - div * 4;
                for (int i = 0; i < rem; ++i)
                {
                    batch_mode[i + normal_size] = 1;
                    batch_shuffle[i + normal_size] = true;
                }

                for (int k = 0; k < 4; ++k)
                {
                    int mode = modes[k];
                    int start = rem + k * div;
                    int end = start + div;
                    for (int i = start; i < end && i < shuffle_size; ++i)
                    {
                        batch_mode[i + normal_size] = mode;
                        batch_shuffle[i + normal_size] = true;
                    }
                }
            }
        }
        else
        {
            int div, rem;
            int normal_size = batch_size;
            if (width == height)
            {
                div = normal_size / 8;
                rem = normal_size - div * 8;
                for (int i = 0; i < rem; ++i)
                    batch_mode[i] = 1;

                for (int mode = 1; mode <= 8; ++mode)
                {
                    int start = rem + (mode - 1) * div;
                    int end = start + div;
                    for (int i = start; i < end && i < normal_size; ++i)
                        batch_mode[i] = mode;
                }
            }
            else
            {
                static int modes[] = {1, 4, 5, 8};

                div = normal_size / 4;
                rem = normal_size - div * 4;
                for (int i = 0; i < rem; ++i)
                    batch_mode[i] = 1;

                for (int k = 0; k < 4; ++k)
                {
                    int mode = modes[k];
                    int start = rem + k * div;
                    int end = start + div;
                    for (int i = start; i < end && i < normal_size; ++i)
                        batch_mode[i] = mode;
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < batch_size; ++i)
        {
            batch_mode[i] = 1;
            batch_shuffle[i] = false;
        }
    }
    
    if (ACTION_PLANES == 2)
    {
        for (int i = 0; i < batch_size; i += 2)
            batch_no_action[i] = true;
    }

    for (int i = 0; i < batch_size; i += 4)
    {
        batch_swap_action[i] = true;
        if (i + 1 < batch_size)
            batch_swap_action[i + 1] = true;
    }

    for (int i = 0; i < log2(batch_size) * SP_THREADS; ++i)
    {
        auto idx = GoRandom::Get().FastR31(batch_size);
        batch_combined[idx] = true;
    }

    for (int i = 0; i < batch_size; ++i)
        transform_data(i, batch_mode[i], batch_shuffle[i], batch_no_action[i], batch_swap_action[i], batch_combined[i]);
}
