//
// Created by yuanyu on 2018.02.01.
//

#include <fstream>
#include "SelfPlay.h"
#include "Random.h"

SelfPlay::SelfPlay(int width, int height, int simulations)
{
    m_width = width;
    m_height = height;
    m_simulations = simulations;
    auto sz = (m_width == m_height) ? to_string(m_width): to_string(m_width) + ":" + to_string(m_height);
    sgf_head = "(;CA[UTF-8]GM[1]FF[4]\nSZ[" + sz + "]";
    sgf_bw = "";
    m_game_number = 1;
    sw_genmove.clear_count();
}

void SelfPlay::run(Network& nn, ReplayPool& pool, int games, bool is_train, bool warm_up)
{
    string sgf_folder = get_sgf_folder(m_width, m_height);
    string data_folder = get_data_folder(m_width, m_height);
    for (auto i = 0; i < games; ++i)
    {
        sgf_bw = "";
        auto next_black = true;
        auto early_stop = false;
        auto ignore_early_stop = false;
        auto early_stop_count = 0;
        auto early_resign = false;
        auto ignore_early_resign = false;
        auto black_resign_count = 0;
        auto white_resign_count = 0;
        auto move_count = 0;

        GoBoard board(m_width, m_height);
        MCTS mcts{board};
        mcts_game.game.clear();
        mcts_game.sample.clear();
        MCTSParam param;
        param.is_train = is_train;

        // is_greedy
        auto board_size = m_width * m_height;
        auto move_count_greedy = 0;
        switch (param.sp_mode)
        {
            case SelfPlayMode::MonteCarlo:
                move_count_greedy = 0;
                break;
            case SelfPlayMode::TreeBackup:
                move_count_greedy = board.move_count_max();
                break;
            case SelfPlayMode::Mixed:
            {
                move_count_greedy = GoRandom::Get().RangeR31(board_size * param.mixed_min, board_size * param.mixed_max);
                break;
            }
        }

        // komi
        if ( !warm_up )
        {
            if (is_train)
                param.simulations = m_simulations;
            else
                param.simulations = min(400, m_simulations / 2);
            mcts.InitVirtualKomi(&nn, param);
            param.simulations = m_simulations;
        }
        else
        {
            param.simulations = m_simulations * 100;
        }

        // selfplay
        while ( !mcts.GameOver() )
        {
            if (early_stop || early_resign)
                break;

            int p = GoBoard::NO_MOVE;
            sw_genmove.start_count();
            param.is_greedy = (move_count >= move_count_greedy) || !param.is_train;
            if ( warm_up )
                p = mcts.UCTGenMove(&nn, param);// 无视SimulationMode, 不记录virtual_win/loss
            else
                p = mcts.GenMove(&nn, param);
            sw_genmove.end_count();

            if ( param.use_early_stop && is_train && !ignore_early_stop )
            {
                if ( mcts.EarlyStop() )
                {
                    ++early_stop_count;
                    if (early_stop_count >= 2)
                    {
                        double r = GoRandom::Get().RFloat();
                        if (r < 0.8)
                            early_stop = true;
                        else
                            ignore_early_stop = true;
                    }
                }
                else
                {
                    early_stop_count = 0;
                }
            }

            if ( param.use_early_resign && is_train && !ignore_early_resign )
            {
                auto resign_result = mcts.EarlyResign();
                if ( resign_result > 0 )
                {
                    black_resign_count += 1;
                    white_resign_count = 0;
                    if (black_resign_count >= 4)
                    {
                        double r = GoRandom::Get().RFloat();
                        if (r < 0.8)
                            early_resign = true;
                        else
                            ignore_early_resign = true;
                    }
                }
                else if ( resign_result < 0 )
                {
                    black_resign_count = 0;
                    white_resign_count += 1;
                    if (white_resign_count >= 4)
                    {
                        double r = GoRandom::Get().RFloat();
                        if (r < 0.8)
                            early_resign = true;
                        else
                            ignore_early_resign = true;
                    }
                }
                else
                {
                    black_resign_count = 0;
                    white_resign_count = 0;
                }
            }

            if (p == GoBoard::PASS_MOVE)
                cerr << endl << "gen move: PASS" << endl;
            else if (p == GoBoard::NO_MOVE)
                break;
            else
                cerr << endl << "gen move: (" << board.posx(p) << "," << board.posy(p) << ")" << endl;

            // sgf
            if ( next_black )
                sgf_bw += ";B[";
            else
                sgf_bw += ";W[";

            if (p != GoBoard::PASS_MOVE)
            {
                int x = board.posx(p);
                int y = board.posy(p);
                sgf_bw += x + 'a';
                sgf_bw += y + 'a';
            }

            sgf_bw += "]";

            // training data
            mcts.CollectData(mcts_game, p, param.is_greedy);

            // Play move
            mcts.Play(p);
            next_black = !next_black;
            ++move_count;
        }
        
        if ( mcts.GameOver() )
        {
            early_stop = false;
            early_resign = false;
        }

        // save
        if (is_train)
        {
            auto final_score = 0;
            if (early_stop)
            {
                final_score = mcts.EarlyStopScore();
                sgf_bw = "RE[" + std::to_string(final_score) + "]" + sgf_bw;
                cerr << "Early Stop" << endl;
            }
            else if (early_resign)
            {
                final_score = mcts.EarlyStopScore();
                sgf_bw = "RE[" + std::to_string(final_score) + "]" + sgf_bw;
                cerr << "Early Resign" << endl;
            }
            else
            {
                final_score = mcts.FinalScore();
                sgf_bw = "RE[" + std::to_string(final_score) + "]" + sgf_bw;
                cerr << "Game Over" << endl;
            }

            auto next_id = next_file_id(data_folder + "/id.txt");
            pool.SaveData(mcts_game, next_id, final_score, m_width, m_height);
            SaveSGF(sgf_folder + "/" + to_string(next_id) + ".SGF", sgf_bw);
            ++m_game_number;

            ToPool(pool);
        }
        else
        {
            cerr << "Game Over" << endl;
        }
    }
}

void SelfPlay::mc_run(Network& nn, ReplayPool& pool, int games, bool is_train)
{
    string sgf_folder = get_sgf_folder(m_width, m_height);
    string data_folder = get_data_folder(m_width, m_height);
    for (int i = 0; i < games; ++i)
    {
        sgf_bw = "";
        bool next_black = true;
        GoBoard board(m_width, m_height);
        MCTS mcts{board};
        mcts_game.game.clear();
        mcts_game.sample.clear();

        while ( !mcts.GameOver() )
        {
            int p = mcts.UCBGenMove(&nn, m_simulations, is_train);
            if (p == GoBoard::PASS_MOVE)
                cerr << endl << "gen move: PASS" << endl;
            else if (p == GoBoard::NO_MOVE)
                break;
            else
                cerr << endl << "gen move: (" << board.posx(p) << "," << board.posy(p) << ")" << endl;

            // sgf
            if ( next_black )
                sgf_bw += ";B[";
            else
                sgf_bw += ";W[";

            if (p != GoBoard::PASS_MOVE)
            {
                int x = board.posx(p);
                int y = board.posy(p);
                sgf_bw += x + 'a';
                sgf_bw += y + 'a';
            }

            sgf_bw += "]";

            // Play move
            mcts.CollectData(mcts_game, p, is_train);
            mcts.Play(p);

            next_black = !next_black;
        }

        if (is_train)
        {
            auto final_score = mcts.FinalScore();

            sgf_bw = "RE[" + std::to_string(final_score) + "]" + sgf_bw;
            cerr << "Game Over" << endl;

            auto next_id = next_file_id(data_folder + "/id.txt");
            pool.SaveData(mcts_game, next_id, final_score, m_width, m_height);
            SaveSGF(sgf_folder + "/" + to_string(next_id) + ".SGF", sgf_bw);
            ++m_game_number;
        }
        else
        {
            cerr << "Game Over" << endl;
        }
    }
}

void SelfPlay::pk(Network& nn, int games, int komi)
{
    bool use_black = true;
    double score = 0.0;
    double win = 0.0;
    double loss = 0.0;
    double draw = 0.0;
    MCTSParam param_b, param_w;
    param_b.simulations = m_simulations;
    param_w.simulations = m_simulations;
    param_b.is_greedy = true;
    param_w.is_greedy = true;
    param_b.puct_mode = PUCTMode::NCK_STDEV;
    param_w.puct_mode = PUCTMode::NCK_STDEV;
    param_b.puct_child_k = 1.5;
    param_w.puct_child_k = 1.25;
    param_b.c_puct = 1.96;
    param_w.c_puct = 1.96;
    for (int i = 0; i < games; ++i)
    {
        sgf_bw = "";
        bool next_black = true;
        GoBoard board(m_width, m_height);
        MCTS mcts_b{board};
        MCTS mcts_w{board};

        while ( !mcts_b.GameOver() )
        {
            int p; 
            if (next_black)
                p = mcts_b.UCTGenMove(&nn, param_b);
            else
                p = mcts_w.UCTGenMove(&nn, param_w);

            if (p == GoBoard::PASS_MOVE)
                cerr << endl << "gen move: PASS" << endl;
            else if (p == GoBoard::NO_MOVE)
                break;
            else
                cerr << endl << "gen move: (" << board.posx(p) << "," << board.posy(p) << ")" << endl;

            // Play move
            if (next_black)
            {
                mcts_w.Play(p, false);
                mcts_b.Play(p);
            }
            else
            {
                mcts_b.Play(p, false);
                mcts_w.Play(p);
            }

            next_black = !next_black;
        }
        
        auto final_score = mcts_b.FinalScore();

        // Win/Draw/Loss
        if (use_black)
        {
            if (final_score > komi)
                ++win;
            else if (final_score < komi)
                ++loss;
            else
                ++draw;
        }
        else
        {
            if (final_score > komi)
                ++loss;
            else if (final_score < komi)
                ++win;
            else
                ++draw;
        }
        
        // Score
        final_score -= komi;
        if ( !use_black )
            final_score = -final_score;
        
        score += final_score;
        use_black = !use_black;
        cerr << "Game Over" << endl;
        cerr << "AVG Score: " << score / (i + 1) << endl;
        cerr << "Win:" << win << " Draw: " << draw << " Loss: " << loss << endl;
        std::swap(param_b, param_w);
    }
}

void SelfPlay::profiler()
{
    sw_genmove.output_count("genmove time");
}

void SelfPlay::SaveSGF(const string& fn, const string& sgf_bw)
{
    ofstream iter_file(fn, ios::out);
    iter_file << sgf_head << sgf_bw << ")";
    cerr << "Saving SGF: " << fn << endl;
}

void SelfPlay::ToPool(ReplayPool& pool)
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

    ReplayGame rg;
    // normal game
    {
        // for early stop
        auto final_board_value = mcts_game.game.back().board_value;
        smooth(final_board_value);
        auto final_value = accumulate(final_board_value.begin(), final_board_value.end(), 0.0f);

        // bv & v
        auto mix_board_value = final_board_value;
        auto mix_value = final_value;
        auto size = int(mcts_game.game.size());
        rg.game.resize(size);
        for (int i = size - 1; i >= 0; --i)
        {
            const auto& data = mcts_game.game[i];
            auto& rn = rg.game[i];
            rn.Init(m_width, m_height, INPUT_HISTORY);
            rn.m_stone = data.stone;
            rn.m_policy = data.policy;
            rn.m_state = data.state;
            rn.m_lr = data.lr;

            if (i != size - 1 && !data.is_greedy)
            {
                // 1. mix
                // bv_{t} = \pi_{t} * bv_{t+1} + (1 - \pi_{t}) * bv^{mcts}{t}
                auto alpha = data.policy[data.np];
                for (size_t k = 0; k < data.board_value.size(); ++k)
                    mix_board_value[k] = alpha * mix_board_value[k] + (1.0 - alpha) * data.board_value[k];

                smooth(mix_board_value);
                mix_value = accumulate(mix_board_value.begin(), mix_board_value.end(), 0.0f);
            }
            // else
            // 2. last data
            // bv_{t} = bv_{T}
            // 3. greedy move
            // bv_{t} = bv_{t+1};

            rn.m_board_value = mix_board_value;
            rn.m_value = mix_value;

            if (data.swap_action)
                rn.m_action = GoBoard::calc_action(rn.m_stone, INPUT_HISTORY, m_width * m_height);
        }
    }

    // sample
    if (USE_RD)
    {
        auto size = int(mcts_game.sample.size());
        rg.sample.resize(size);
        for (int i = 0; i < size; ++i)
        {
            const auto& data = mcts_game.sample[i];
            auto& rn = rg.sample[i];
            rn.Init(m_width, m_height, INPUT_HISTORY);
            rn.m_stone = data.stone;
            rn.m_policy = data.policy;
            rn.m_state = data.state;
            rn.m_lr = data.lr;

            rn.m_board_value = data.board_value;
            smooth(rn.m_board_value);
            rn.m_value = accumulate(rn.m_board_value.begin(), rn.m_board_value.end(), 0.0f);

            if (data.swap_action)
                rn.m_action = GoBoard::calc_action(rn.m_stone, INPUT_HISTORY, m_width * m_height);
        }
    }

    pool.Put(rg);
}
