#include <chrono>
#include <fstream>
#include "Network.h"
#include "GoBoard.h"
#include "MCTS.h"
#include "Random.h"
#include "ReplayPool.h"
#include "Utils.h"
#include "SelfPlay.h"
#include "Zobrist.h"
#include "config.h"

using namespace std;

void Init(int board_width, int board_height)
{
    GoHash.Init(board_width, board_height);
}

void MCTS_Train(int board_width = 3, int board_height = 3, int games = 3000)
{
    auto board_size = board_width * board_height;
    auto simulations = max(20 / SP_THREADS, 5) * board_size;
    auto buffer_game_size = int(pow(board_size, 1.75) * SP_THREADS);
    //auto buffer_game_size = max(board_size, SP_THREADS) * 20;
    folder_init(board_width, board_height);
    GoHash.Init(board_width, board_height);
    ReplayPool pool(buffer_game_size);// 1,000,000;
    Network nn(board_width, board_height);
    SelfPlay sp(board_width, board_height, simulations);
    StopWatch sw_train, sw_total;
    sw_train.clear_count();
    sw_total.clear_count();

    // load train data
    pool.LoadData(board_width, board_height, buffer_game_size);
    auto pool_size = pool.DataSize();

    sw_total.start_count();
    auto batch_size = 1024;
    auto iter = int(board_size * ITER_RATE);

    // uct+rollout self-play
    int start = 1;
    if ( nn.is_empty_model() )
    {
        int warm_up_games = 32;
        for (int i = 1; i <= warm_up_games; ++i)
        {
            cerr << "warm up, train:" << i << endl;
            sp.run(nn, pool, 1, true, true);
        }
        sw_train.start_count();
        nn.train(pool, iter * warm_up_games, batch_size);// AGZ: 1000, 2048
        sw_train.end_count();
        start += warm_up_games;
    }

    // nn self-play
    for (int i = start; i <= games; ++i)
    {
        cerr << "self-play, train:" << i << endl;
        sp.run(nn, pool, 1);

        auto total_games = pool_size + i;
        bool train = (total_games % SP_THREADS == 0) && total_games >= 8 * SP_THREADS;
        if (train)
        {
            sw_train.start_count();
            nn.train(pool, iter, batch_size);// AGZ: 1000, 2048
            sw_train.end_count();
        }
    }
    sw_total.end_count();

    // output
    sw_train.output_count("train time");
    sp.profiler();
    nn.stopwatch_output();
    sw_total.output_count("total time");
}

void MC_Train(int board_width = 3, int board_height = 3, int games = 1, int simulations = 20000)
{
    folder_init(board_width, board_height);
    GoHash.Init(board_width, board_height);
    int board_size = board_width * board_height;
    ReplayPool pool(1);// 1,000,000;
    Network nn(board_width, board_height);
    SelfPlay sp(board_width, board_height, simulations);
    StopWatch sw_train, sw_total;
    sw_train.clear_count();
    sw_total.clear_count();

    sw_total.start_count();
    sp.mc_run(nn, pool, games, false);
    sw_total.end_count();
}

void Trace_Test(int board_width = 4, int board_height = 3, int games = 1, int simulations = 20000, int komi = 4)
{
    folder_init(board_width, board_height);
    GoHash.Init(board_width, board_height);
    int board_size = board_width * board_height;
    ReplayPool pool(1);// 1,000,000;
    Network nn(board_width, board_height);
    SelfPlay sp(board_width, board_height, simulations);
    StopWatch sw_train, sw_total;
    sw_train.clear_count();
    sw_total.clear_count();

    sw_total.start_count();
    sp.pk(nn, games, komi);
    sw_total.end_count();
}

#include <random>
vector<float> dir(101, 0);
void noise()
{
    //auto size = GoRandom::Get().FastRange31(3, 20);
    auto size = 3;
    auto child_count = size * size;
    float alpha = 1000.0f / child_count;

    auto dirichlet_vector = std::vector<float>{};
    std::gamma_distribution<float> gamma(alpha, 1.0f);

    for (size_t i = 0; i < child_count; i++)
        dirichlet_vector.emplace_back(gamma(GoRandom::Get()));

    auto sample_sum = std::accumulate(begin(dirichlet_vector), end(dirichlet_vector), 0.0f);
    if ( sample_sum < std::numeric_limits<float>::min() )
        return;

    for (auto& v: dirichlet_vector)
        v /= sample_sum;

    sort(dirichlet_vector.begin(), dirichlet_vector.end(), [](float a, float b){ return a > b; });
    for (auto& v: dirichlet_vector)
        ++dir[int(v * 100)];
}

#include "Symbol.h"
void SaveSymbol()
{
    auto planes = Symbol::Variable("data");
    auto policy_mcts = Symbol::Variable("mcts_p");
    auto board_value_mcts = Symbol::Variable("mcts_bv");
    auto next_state = Symbol::Variable("next_state");
    auto lr = Symbol::Variable("mcts_lr");
    // auto ResNet = ResNetV2Symbol(planes, policy_mcts, board_value_mcts, next_state, lr, 64, 2, 361, false, "relu", "kt2");
    auto ResNet = ResNetV2Output(planes, 64, 2, false, "relu", "bn", "kt.b");

    ResNet.Save("res.txt");
    //auto sym = IterMixNetSymbol(planes, policy_mcts, board_value_mcts, next_state, lr, 16, 2, 6, 10, "");
    //sym.Save("sym.txt");
}

void Data_Train(int board_width, int board_height, int buffer_size, int epoch)
{
    folder_init(board_width, board_height);
    GoHash.Init(board_width, board_height);
    ReplayPool pool(buffer_size);// 1,000,000;
    Network nn(board_width, board_height);
    StopWatch sw_train, sw_total;
    SelfPlay sp(board_width, board_height, 400);
    sw_train.clear_count();
    sw_total.clear_count();
    sw_total.start_count();
    pool.LoadData(board_width, board_height, buffer_size);

    //for (int i = 1; i <= 5; ++i)
    {
        nn.train(pool, epoch, 128);
        ///nn.test_predict(pool, 3);
        //sp.run(nn, pool, 2);
    }

    sw_total.end_count();
    sw_total.output_count("total");
}

void Self_Play_Test(int board_width, int board_height, int games, int playout)
{
    folder_init(board_width, board_height);
    GoHash.Init(board_width, board_height);
    ReplayPool pool(1);// 1,000,000;
    Network nn(board_width, board_height);
    StopWatch sw_train, sw_total;
    SelfPlay sp(board_width, board_height, playout);
    sp.run(nn, pool, games, false);
}

void Net2Net(int new_blocks, int new_filters, string se_type)
{
    Network nn(9, 9);
    nn.net2net(new_blocks, new_filters, se_type);
}

void MXVersion()
{
    int version = 0;
    MXGetVersion(&version);
    cerr << "MXNet Version: " << version << endl;
}

#include <dmlc/parameter.h>
#include <dmlc/timer.h>
void TestCode()
{
    //MXVersion();
    auto start = dmlc::GetTime();
    cerr << dmlc::GetEnv("MXNET_CPU_WORKER_NTHREADS", 0) << endl;
    fprintf(stderr, "%f\n", dmlc::GetTime());
    cerr << dmlc::GetTime() - start << endl;
    cerr << __FILE__ << __LINE__ << endl;
}

int main(int argc, char** argv)
{
    //SaveSymbol();
    //return 0;
    MXVersion();
    if (atoi(argv[1]) == 5)
    {
        int width = 4;
        int height = 3;
        int games = 1000;
        int playout = 20000;
        int komi = 4;
        if (argc == 7)
        {
            width = atoi(argv[2]);
            height = atoi(argv[3]);
            games = atoi(argv[4]);
            playout = atoi(argv[5]);
            komi = atoi(argv[6]);
        }
        Trace_Test(width, height, games, playout, komi);
    }
    else if (atoi(argv[1]) == 4)
    {
        int new_blocks = 4;
        int new_filters = 128;
        string se_type = "";
        if (argc == 5)
        {
            new_blocks = atoi(argv[2]);
            new_filters = atoi(argv[3]);
            se_type = string(argv[4]);
        }
        Net2Net(new_blocks, new_filters, se_type);
    }
    else if (atoi(argv[1]) == 3)
    {
        int width = 7;
        int height = 7;
        int games = 1;
        int playout = 1;
        if (argc == 6)
        {
            width = atoi(argv[2]);
            height = atoi(argv[3]);
            games = atoi(argv[4]);
            playout = atoi(argv[5]);
        }
        MC_Train(width, height, games, playout);
    }
    else if (atoi(argv[1]) == 2)
    {
        int width = 3;
        int height = 3;
        int buffer_size = 1;
        int epoch = 1000;
        if (argc == 5)
        {
            width = atoi(argv[2]);
            height = atoi(argv[3]);
            epoch = atoi(argv[4]);
        }
        MCTS_Train(width, height, epoch);
    }
    else if (atoi(argv[1]) == 1)
    {
        int width = 3;
        int height = 3;
        int buffer_size = 1;
        int epoch = 1000;
        if (argc == 6)
        {
            width = atoi(argv[2]);
            height = atoi(argv[3]);
            buffer_size = atoi(argv[4]);
            epoch = atoi(argv[5]);
        }
        Data_Train(width, height, buffer_size, epoch);
    }
    else
    {
        int width = 7;
        int height = 7;
        int games = 1;
        int playout = 1;
        if (argc == 6)
        {
            width = atoi(argv[2]);
            height = atoi(argv[3]);
            games = atoi(argv[4]);
            playout = atoi(argv[5]);
        }
        Self_Play_Test(width, height, games, playout);
    }

    //MC_Train(width, height, epoch);
    //SaveSymbol();

    //for (int i = 0; i < 10000; ++i)
    //    noise();

    //int t = 0;
    return 0;
}
