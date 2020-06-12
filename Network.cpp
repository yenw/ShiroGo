//
// Created by yuanyu on 2018.01.17.
//
#include <iostream>
#include <fstream>
#include <random>
#include "Network.h"
#include "NNMetric.h"
#include "Symbol.h"
#include "Random.h"
#include "config.h"
using namespace std;

void Network::reset_ctx(std::map<std::string, NDArray>& args_map)
{
    for (auto& iter: args_map)
        args_map[iter.first] = iter.second.Copy(ctx);
}

void Network::weight_init(std::map<std::string, NDArray>& args_map, std::map<std::string, NDArray>& aux_map)
{
    if (ACT_TYPE == "selu")
    {
        cerr << "Lecun Normal" << endl;
        auto xavier = Xavier(Xavier::gaussian, Xavier::in, 1.0f);
        for (auto &arg: args_map)
            xavier(arg.first, &arg.second);

        //for (auto &aux: aux_map)
        //    xavier(aux.first, &aux.second);
    }
    else
    {
        cerr << "Xavier" << endl;
        auto xavier = Xavier(Xavier::gaussian, Xavier::avg, 3.0f);
        for (auto &arg: args_map)
            xavier(arg.first, &arg.second);

        //for (auto &aux: aux_map)
        //    xavier(aux.first, &aux.second);
    }
}

void Network::weight_load(std::map<std::string, NDArray>& args_map, std::map<std::string, NDArray>& aux_map)
{
    string load_path = folder_model + "/weight_" + to_string(train_iter % 10) + ".param";
    cerr << "Loading from " << load_path << endl;

    std::map<std::string, NDArray> params = NDArray::LoadToMap(load_path);
    reset_ctx(params);
    for (auto iter : params)
    {
        string type = iter.first.substr(0, 4);
        string name = iter.first.substr(4);
        NDArray target;
        if (type == "arg:")
            args_map.insert({name, iter.second});
        else if (type == "aux:")
            aux_map.insert({name, iter.second});
    }
}

void Network::weight_save(Executor* exec)
{
    std::map<std::string, NDArray> params;
    for (auto arg: exec->arg_dict())
    {
        if ( is_variable(arg.first) )
            continue;

        params.insert({"arg:" + arg.first, arg.second});
    }

    for (auto aux: exec->aux_dict())
        params.insert({"aux:" + aux.first, aux.second});

    string save_path = folder_model + "/weight_" + to_string(train_iter % 10) + ".param";
    cerr << "Saving to " << save_path << endl;
    NDArray::Save(save_path, params);
}

void Network::config_save()
{
    ofstream iter_file(folder_model + "/config.txt", ios::out);
    iter_file << (train_iter);
    cerr << "Save Weight ID: " << train_iter << endl;
}

void Network::config_load()
{
    train_iter = 0;
    predict_rand = true;
    ifstream iter_file(folder_model + "/config.txt");
    string temp;
    if ( iter_file.is_open() )
    {
        iter_file >> train_iter;
        cerr << "Load Weight ID: " << train_iter << endl;

        predict_rand = (train_iter == 0);// 加载时判断predict是否使用随机策略.
    }
}

void Network::arg_map_init(std::map<std::string, NDArray>& args_map, size_t init_batch_size)
{
    args_map["data"] = NDArray(Shape(init_batch_size, input_planes, img_height, img_width), ctx);
    args_map["mcts_p"] = NDArray(Shape(init_batch_size, img_size + 1), ctx);
    args_map["mcts_bv"] = NDArray(Shape(init_batch_size, img_size), ctx);
    args_map["next_state"] = NDArray(Shape(init_batch_size, img_size * 3), ctx);
    args_map["mcts_lr"] = NDArray(Shape(init_batch_size, 1), ctx);
}

bool Network::is_variable(const std::string& name)
{
    if (name == "data")
        return true;

    if (name == "mcts_p")
        return true;

    if (name == "mcts_bv")
        return true;
    
    if (name == "next_state")
        return true;
    
    if (name == "mcts_lr")
        return true;
    
    return false;
}

Network::Network(uint32_t width, uint32_t height)
{
    // ctx
    MXGetGPUCount(&num_gpu);
    if (num_gpu > 0)
        ctx = Context::gpu();

    body_blocks = 4;
    body_filters = 64;

    batch_size = 128;
    predict_batch_size = 1;
    predict_exec = nullptr;
    train_opt = nullptr;

    // board size
    img_width = width;
    img_height = height;
    img_size = width * height;

    // network
    input_planes = INPUT_HISTORY * 2 + COLOR_PLANES + ACTION_PLANES;

    sw_predict.clear_count();
    folder_model = get_model_folder();

    config_load();
}

Network::~Network()
{
    delete predict_exec;
    delete train_opt;
    MXNotifyShutdown();
}

Symbol Network::GetSymbol(uint32_t tower_size, uint32_t num_filter, bool is_train)
{
    auto planes = Symbol::Variable("data");
    auto policy_mcts = Symbol::Variable("mcts_p");
    auto board_value_mcts = Symbol::Variable("mcts_bv");
    auto next_state = Symbol::Variable("next_state");
    auto lr = Symbol::Variable("mcts_lr");
    auto iter = std::max(img_width, img_height);
    auto bottle_neck = false;
    auto norm_type = "bn";
    auto se_type = "kt.b";

    if (is_train)
        return ResNetV2Symbol(planes, policy_mcts, board_value_mcts, next_state, lr, num_filter, tower_size, img_size, bottle_neck, ACT_TYPE, norm_type, se_type);
    else
        return ResNetV2Output(planes, num_filter, tower_size, bottle_neck, ACT_TYPE, norm_type, se_type);
    //return ResNetCZSymbol(planes, policy_mcts, board_value_mcts, num_filter, tower_size, img_size, false, "relu", "");
    //return DPNSymbol(planes, policy_mcts, board_value_mcts, num_filter, tower_size, img_size, 8);
    //return MixNetSymbol(planes, policy_mcts, board_value_mcts, num_filter, tower_size, img_size, true, "relu", "");
    //return IterMixNetSymbol(planes, policy_mcts, board_value_mcts, num_filter, tower_size, img_size, iter, "");
}

void Network::train(ReplayPool& pool, uint32_t iters, uint32_t mini_batch)
{
    delete predict_exec;
    predict_exec = nullptr;

    auto train_batch_size = batch_size;
    if (mini_batch != 0)
        train_batch_size = mini_batch;

    Symbol model = GetSymbol(body_blocks, body_filters);
    Executor* exec;
    std::map<string, NDArray> args_map, aux_map;
    arg_map_init(args_map, train_batch_size);

    if (train_iter != 0)
    {
        std::map<std::string, NDArray> arg_grad_store;
        std::map<std::string, OpReqType> grad_req_type;

        weight_load(args_map, aux_map);
        exec = model.SimpleBind(ctx, args_map, arg_grad_store, grad_req_type, aux_map);
    }
    else
    {
        //以下代码在1.4.1可以运行, 在1.6最后会输出nan
        //1.6版之所以输出nan, 可能是因为未经训练, 直接使用moving_mean/moving_var;
        exec = model.SimpleBind(ctx, args_map);
        args_map = exec->arg_dict();
        aux_map = exec->aux_dict();
        weight_init(args_map, aux_map);
    }

    if (train_opt == nullptr)
    {
        // train_opt 的 num_update 是不正确的.
        // 不过 sgd_mem 也没用上 num_update.
        // 但是 adam 需要 num_update, 需要注意.
        //
        // momentum 的参数保存在c++ api 中.
        // 一旦delete train_opt 则 momentum 就清空了.
        // 而c++ api也没有保存 momentum 的函数, 清空也没办法.
        train_opt = OptimizerRegistry::Find("sgd");

        train_opt->SetParam("rescale_grad", 1.0 / train_batch_size);

        if (ACT_TYPE == "selu")
            train_opt->SetParam("lr", 0.01);
        else
            train_opt->SetParam("lr", 0.1);

        train_opt->SetParam("momentum", 0.9);
        train_opt->SetParam("wd", 0.0001);
        //train_opt->SetParam("clip_gradient", 10);
    }

    vector<mx_float> board_batch;
    vector<mx_float> policy_batch;
    vector<mx_float> board_value_batch;
    vector<mx_float> board_state_batch;
    vector<mx_float> lr_batch;

    BV_MSE metric_bv_mse;// Board Value
    BV_CE metric_bv_ce;
    PI_MSE metric_p_mse;// Policy
    PI_CE metric_p_ce;
    S_MSE metric_s_mse;// State
    S_CE metric_s_ce;
    NN_LOSS metric_loss;
    StopWatch sw_log;
    auto arg_names = model.ListArguments();
    auto metric_size = int(img_size * ITER_RATE * 0.5);
    sw_log.start();
    for (int iter = 0; iter < iters; ++iter)
    {
        pool.GetBatch(board_batch, policy_batch, board_value_batch, board_state_batch, lr_batch, train_batch_size);
        auto data = NDArray(board_batch, Shape(train_batch_size, input_planes, img_height, img_width), ctx);
        auto label_p = NDArray(policy_batch, Shape(train_batch_size, img_size + 1), ctx);
        auto label_bv = NDArray(board_value_batch, Shape(train_batch_size, img_size), ctx);
        auto label_state = NDArray(board_state_batch, Shape(train_batch_size, img_size * 3), ctx);
        auto label_lr = NDArray(lr_batch, Shape(train_batch_size, 1), ctx);

        data.CopyTo(&exec->arg_dict()["data"]);
        label_p.CopyTo(&exec->arg_dict()["mcts_p"]);
        label_bv.CopyTo(&exec->arg_dict()["mcts_bv"]);
        label_state.CopyTo(&exec->arg_dict()["next_state"]);
        label_lr.CopyTo(&exec->arg_dict()["mcts_lr"]);
        NDArray::WaitAll();
        exec->Forward(true);

        if ( (iter + 1) % metric_size == 0 )
        {
            NDArray::WaitAll();
            cerr << "  update: " << to_string(train_iter) << " -> (" << to_string(iter + 1)
                 << "/" << to_string(iters + 1) << ")" << endl;

            // MSE
            auto local_p_mse = metric_p_mse.UpdateMetric(label_p, exec->outputs[0], train_batch_size);
            cerr << "   p_mse: " << metric_p_mse.Get() << endl;
            cerr << " t_p_mse: " << local_p_mse / train_batch_size << endl;

            auto local_bv_mse = metric_bv_mse.UpdateMetric(label_bv, exec->outputs[1], train_batch_size, img_size);
            cerr << "  bv_mse: " << metric_bv_mse.Get() << endl;
            cerr << "t_bv_mse: " << local_bv_mse / train_batch_size << endl;

            auto local_s_mse = metric_s_mse.UpdateMetric(label_state, exec->outputs[2], train_batch_size, img_size);
            cerr << "   s_mse: " << metric_s_mse.Get() << endl;
            cerr << " t_s_mse: " << local_s_mse / train_batch_size << endl;

            // Cross Entropy
            auto local_p_ce = metric_p_ce.UpdateMetric(label_p, exec->outputs[0], train_batch_size);
            cerr << "    p_ce: " << metric_p_ce.Get() << endl;
            cerr << "  t_p_ce: " << local_p_ce / train_batch_size << endl;

            auto local_bv_ce = metric_bv_ce.UpdateMetric(label_bv, exec->outputs[1], train_batch_size, img_size);
            cerr << "   bv_ce: " << metric_bv_ce.Get() << endl;
            cerr << " t_bv_ce: " << local_bv_ce / train_batch_size << endl;

            auto local_s_ce = metric_s_ce.UpdateMetric(label_state, exec->outputs[2], train_batch_size, img_size);
            cerr << "    s_ce: " << metric_s_ce.Get() << endl;
            cerr << "  t_s_ce: " << local_s_ce / train_batch_size << endl;

            // Loss
            auto local_loss = metric_loss.UpdateMetric(exec->outputs[3], train_batch_size);
            cerr << "    loss: " << metric_loss.Get() << endl;
            cerr << "  t_loss: " << local_loss / train_batch_size << endl;
            cerr << "---------" << endl;
        }
        else if ( sw_log.timeout(5.0) )
        {
            cerr << " timeout: " << to_string(train_iter) << " -> (" << to_string(iter + 1)
                 << "/" << to_string(iters + 1) << ")" << endl;
            cerr << "---------" << endl;
            sw_log.start();
        }

        exec->Backward();
        for (size_t i = 0; i < arg_names.size(); ++i)
        {
            if ( is_variable(arg_names[i]) )
                continue;

            train_opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
        }

        NDArray::WaitAll();
    }

    predict_rand = false;// 已经完成训练, 通知predict加载模型, 禁用随机策略
    ++train_iter;
    weight_save(exec);
    config_save();
    delete exec;
}

void Network::predict(vector<float>& board_batch, vector<float>& policy_batch, vector<float>& board_value_batch)
{
    sw_predict.start_count();
    if ( !predict_rand )
    {
        if (predict_exec == nullptr)
        {
            // 重新加载配置
            config_load();
            assert(predict_rand == false);

            // 初始化model, args
            auto model = GetSymbol(body_blocks, body_filters, false);
            std::map<string, NDArray> args_map, aux_map;
            arg_map_init(args_map, predict_batch_size);

            // 加载weight
            std::map<std::string, NDArray> arg_grad_store;
            std::map<std::string, OpReqType> grad_req_type;

            weight_load(args_map, aux_map);
            predict_exec = model.SimpleBind(ctx, args_map, arg_grad_store, grad_req_type, aux_map);
        }

        // predict
        auto data = NDArray(board_batch, Shape(predict_batch_size, input_planes, img_height, img_width), ctx);
        data.CopyTo(&predict_exec->arg_dict()["data"]);
        NDArray::WaitAll();
        predict_exec->Forward(false);
        NDArray::WaitAll();
        auto out_policy = predict_exec->outputs[0];
        auto out_value = predict_exec->outputs[1];
        out_policy.SyncCopyToCPU(&policy_batch);
        out_value.SyncCopyToCPU(&board_value_batch);
    }
    else// 均匀策略 & 均匀回报
    {
        // 以下代码被warm_up替代.
        policy_batch.resize(predict_batch_size * (img_size + 1));
        board_value_batch.resize(predict_batch_size * img_size, 0.0f);

        // policy
        auto default_policy = 1.0f / (img_size + 1);
        auto dir_alpha = 10.0f;
        std::vector<float> dir_noise;
        for (int i = 0; i < predict_batch_size; ++i)
        {
            dir_noise = dirichlet_dist(dir_alpha, img_size + 1);
            auto* policy = &policy_batch[i * (img_size + 1)];
            for (int k = 0; k < img_size + 1; ++k)
                policy[k] = dir_noise[k] * 0.25f + default_policy * 0.75f;
        }

        // board value
        auto noise = 0.1f;
        for (auto& v: board_value_batch)
            v = (GoRandom::Get().RFloat() * 2.0f - 1.0f) * noise;
    }

    sw_predict.end_count();
}

std::vector<float> Network::dirichlet_dist(float alpha, int size)
{
    while (true)
    {
        auto dirichlet_vector = std::vector<float>{};
        std::gamma_distribution<float> gamma(alpha, 1.0f);

        for (size_t i = 0; i < size; ++i)
            dirichlet_vector.emplace_back(gamma(GoRandom::Get()));

        auto sample_sum = std::accumulate(begin(dirichlet_vector), end(dirichlet_vector), 0.0f);
        if ( sample_sum > std::numeric_limits<float>::min() )
        {
            for (auto& v: dirichlet_vector)
                v /= sample_sum;

            return dirichlet_vector;
        }
    }
}

void Network::n2n_weight_save(const std::map<std::string, NDArray>& args_map, const std::map<std::string, NDArray>& aux_map)
{
    std::map<std::string, NDArray> params;
    for (auto arg: args_map)
    {
        if ( is_variable(arg.first) )
            continue;

        params.insert({"arg:" + arg.first, arg.second});
    }

    for (auto aux: aux_map)
        params.insert({"aux:" + aux.first, aux.second});

    reset_ctx(params);
    string save_path = folder_model + "/weight_" + to_string(train_iter % 10) + ".param";
    cerr << "N2N: Saving to " << save_path << endl;
    NDArray::Save(save_path, params);
}

void Network::n2n_weight_load(std::map<std::string, NDArray>& args_map, std::map<std::string, NDArray>& aux_map)
{
    string load_path = folder_model + "/weight_" + to_string(train_iter % 10) + ".param.bak";
    cerr << "N2N: Loading from " << load_path << endl;

    std::map<std::string, NDArray> params = NDArray::LoadToMap(load_path);
    for (auto iter : params)
    {
        string type = iter.first.substr(0, 4);
        string name = iter.first.substr(4);
        NDArray target;
        if (type == "arg:")
            args_map.insert({name, iter.second});
        else if (type == "aux:")
            aux_map.insert({name, iter.second});
    }
}

void Network::n2n_weight_deeper(std::map<std::string, NDArray>& args_map, std::map<std::string, NDArray>& aux_map,
                                uint32_t old_blocks, uint32_t new_blocks, uint32_t new_filters, std::string se_type,
                                float noise)
{
    if (new_blocks < old_blocks)
    {
        cerr << "new blocks < old blocks" << endl;
        return;
    }

    if (new_blocks == old_blocks)
        return;

    // args
    // conv.w: (output_channels, input_channels, height, width)
    // conv.b: (output_cnannels)
    // bn.gamma: (output_cnannels)
    // bn.beta: (output_cnannels)

    // aux
    // bn.moving_mean: (output_cnannels)
    // bn.moving_var: (output_cnannels)

    auto noise_init = Uniform(noise);
    auto n2n_conv_noise = [&noise_init](const Shape& shape)
    {
        auto conv = NDArray(shape, Context::cpu());
        noise_init("", &conv);
        return conv;
    };

    std::transform(se_type.begin(), se_type.end(), se_type.begin(), ::tolower);
    auto shape_conv3x3 = Shape(new_filters, new_filters, 3, 3);
    auto shape_conv1x1_se = Shape(new_filters, new_filters * 2, 1, 1);
    auto shape_1D = Shape(new_filters);
    for (auto block = old_blocks; block < new_blocks; ++block)
    {
        auto prefix = "res_" + to_string(block + 1);
        for (auto layer = 1; layer <= 2; ++layer)
        {
            auto l = to_string(layer);
            auto conv_prefix = prefix + "_conv" + l;
            auto bn_prefix = prefix + "_bn" + l;

            // bn: ((w - m_mean) / m_var) * gamma + beta
            aux_map.insert({bn_prefix + "_moving_mean", NDArray(shape_1D, Context::cpu()) = 0});
            aux_map.insert({bn_prefix + "_moving_var", NDArray(shape_1D, Context::cpu()) = 1});

            args_map.insert({bn_prefix + "_gamma", NDArray(shape_1D, Context::cpu()) = 1});
            args_map.insert({bn_prefix + "_beta", NDArray(shape_1D, Context::cpu()) = 0});

            // conv: w * x + b
            args_map.insert({conv_prefix + "_3x3_w", n2n_conv_noise(shape_conv3x3)});
            args_map.insert({conv_prefix + "_3x3_b", NDArray(shape_1D, Context::cpu()) = 0});
        }

        // kt, kt.b
        auto se_prefix = prefix + "_" + se_type;
        if (se_type == "kt" || se_type == "kt.b")
        {
            auto conv_prefix = se_prefix + "_conv";
            auto bn_prefix = se_prefix + "_bn";

            // bn: ((w - m_mean) / m_var) * gamma + beta
            aux_map.insert({bn_prefix + "_moving_mean", NDArray(shape_1D, Context::cpu()) = 0});
            aux_map.insert({bn_prefix + "_moving_var", NDArray(shape_1D, Context::cpu()) = 1});

            args_map.insert({bn_prefix + "_gamma", NDArray(shape_1D, Context::cpu()) = 1});
            args_map.insert({bn_prefix + "_beta", NDArray(shape_1D, Context::cpu()) = 0});

            // conv: w * x + b
            args_map.insert({conv_prefix + "_1x1_w", n2n_conv_noise(shape_conv1x1_se)});
            args_map.insert({conv_prefix + "_1x1_b", NDArray(shape_1D, Context::cpu()) = 0});
        }
    }
}

void Network::n2n_weight_wider(std::map<std::string, NDArray>& args_map, std::map<std::string, NDArray>& aux_map,
                               uint32_t old_blocks, uint32_t old_filters, uint32_t new_filters, std::string se_type,
                               float noise, float dir_alpha)
{
    if (new_filters < old_filters)
    {
        cerr << "new filters < old filters" << endl;
        return;
    }

    if (new_filters == old_filters)
        return;

    auto new_remap = [old_filters, new_filters]()
    {
        vector<int> remap(new_filters);
        for (int i = 0; i < old_filters; ++i)
            remap[i] = i;

        for (int i = old_filters; i < new_filters; ++i)
        {
            auto idx = GoRandom::Get().RangeR31(0, old_filters);
            remap[i] = idx;
        }

        return remap;
    };

    vector<int> global_filter_remap = new_remap();
    vector<int> nn_filter_remap;

    // copy: bias
    auto copy_1D = [&](const string& name)
    {
        args_map[name] = args_map[name].Copy(Context::cpu());
    };
    
    // n2n: bias, bn
    auto n2n_1D = [&](const string& name, bool is_aux)
    {
        vector<mx_float> old_data;
        if (is_aux)
            aux_map[name].SyncCopyToCPU(&old_data);
        else
            args_map[name].SyncCopyToCPU(&old_data);

        auto new_data = std::vector<mx_float>(new_filters, 0.0f);
        for (int i = 0; i < new_filters; ++i)
            new_data[i] = old_data[nn_filter_remap[i]];

        auto new_nd = NDArray(new_data, Shape(new_filters), Context::cpu());
        if (is_aux)
            aux_map[name] = new_nd;
        else
            args_map[name] = new_nd;
    };

    auto n2n_bn = [&](const string& name)
    {
        n2n_1D(name + "_moving_mean", true);
        n2n_1D(name + "_moving_var", true);
        n2n_1D(name + "_gamma", false);
        n2n_1D(name + "_beta", false);
    };

    // n2n: conv3x3, conv1x1
    auto pos = [](int out, int in, int h, int w, vector<mx_uint>& shape)
    {
        return w + shape[3] * (h + shape[2] * (in + shape[1] * out));
    };

    auto noise_init = Uniform(noise);
    auto n2n_conv = [&](const string& name, const string& remap_type, bool n2n_input, bool n2n_output, bool se = false)
    {
        vector<mx_float> data;
        auto nd = args_map[name];
        auto shape = nd.GetShape();
        nd.SyncCopyToCPU(&data);

        auto get_noise = [&noise_init](const vector<mx_uint>& shape)
        {
            auto rate_nd = NDArray(shape, Context::cpu());
            noise_init("", &rate_nd);
            vector<mx_float> rate;
            rate_nd.SyncCopyToCPU(&rate);
            return rate;
        };

        auto new_scale = [this, old_filters, new_filters](const vector<int>& remap, float dir_alpha)
        {
            vector<uint32_t> count(old_filters, 0);
            for (int i = 0; i < new_filters; ++i)
                ++count[remap[i]];

            vector<float> scale(new_filters, 0.0f);

            if (dir_alpha == 0.0f)
            {
                for (int i = 0; i < new_filters; ++i)
                    scale[i] = 1.0f / count[remap[i]];
            }
            else
            {
                for (int i = 0; i < old_filters; ++i)
                {
                    auto dir_scale = dirichlet_dist(dir_alpha, count[i]);
                    for (int k = 0, next = 0; k < new_filters; ++k)
                    {
                        if (remap[k] == i)
                        {
                            scale[k] = dir_scale[next];
                            ++next;
                        }
                    }
                }
            }

            return scale;
        };

        if (n2n_input)// (64, old, 3, 3) -> (64, new, 3, 3)
        {
            auto new_shape = shape;
            auto se_scale = 1.0f;
            if (se)// se_type = kt/kt2
            {
                new_shape[1] = new_filters * (shape[1] / shape[0]);// (64, old * k, 1, 1) -> (64, new * k, 1, 1)
                se_scale = float(shape[0]) / shape[1];
            }
            else
                new_shape[1] = new_filters;// (64, old, 3, 3) -> (64, new, 3, 3)

            vector<mx_float> new_data(new_shape[0] * new_shape[1] * new_shape[2] * new_shape[3], 0.0f);
            vector<mx_float> rate = get_noise(shape);
            vector<mx_float> filter_scale = new_scale(nn_filter_remap, dir_alpha);

            for (int out = 0; out < new_shape[0]; ++out)
            {
                for (int in = 0; in < new_shape[1]; ++in)
                {
                    for (int h = 0; h < new_shape[2]; ++h)
                    {
                        for (int w = 0; w < new_shape[3]; ++w)
                        {
                            auto new_pos = pos(out, in, h, w, new_shape);
                            auto old_pos = pos(out, nn_filter_remap[in % new_filters], h, w, shape);// % for se_type

                            // Leela n2n: rate = (in > shape[1]) ? rate: 0;
                            new_data[new_pos] = data[old_pos] * (1.0f + rate[old_pos]) * filter_scale[in % new_filters] * se_scale;// % for se_type
                        }
                    }
                }
            }

            shape.swap(new_shape);
            data.swap(new_data);
        }

        if (n2n_output)// (old, 18, 3, 3) -> (new, 18, 3, 3)
        {
            if (remap_type == "random")
                nn_filter_remap = new_remap();
            else
                nn_filter_remap = global_filter_remap;

            auto new_shape = shape;
            new_shape[0] = new_filters;
            vector<mx_float> new_data(new_shape[0] * new_shape[1] * new_shape[2] * new_shape[3], 0.0f);
            vector<mx_float> rate = get_noise(shape);

            for (int out = 0; out < new_shape[0]; ++out)
            {
                for (int in = 0; in < new_shape[1]; ++in)
                {
                    for (int h = 0; h < new_shape[2]; ++h)
                    {
                        for (int w = 0; w < new_shape[3]; ++w)
                        {
                            auto new_pos = pos(out, in, h, w, new_shape);
                            auto old_pos = pos(nn_filter_remap[out], in, h, w, shape);

                            // Leela n2n: rate = 0;
                            new_data[new_pos] = data[old_pos] * (1.0f + rate[old_pos]);
                        }
                    }
                }
            }

            shape.swap(new_shape);
            data.swap(new_data);
        }

        args_map[name] = NDArray(data, Shape(shape), Context::cpu());
    };

    // Visual
    auto print_conv3x3 = [&](const string& name)
    {
        vector<mx_float> data;
        auto nd = args_map[name];
        auto shape = nd.GetShape();
        nd.SyncCopyToCPU(&data);

        for (int out = 0; out < shape[0]; ++out)
        {
            cerr << "conv: " << out << endl;
            for (int in = 0; in < shape[1]; ++in)
            {
                cerr << "  filter: " << in << endl;
                auto d00 = data[pos(out, in, 0, 0, shape)];
                auto d01 = data[pos(out, in, 0, 1, shape)];
                auto d02 = data[pos(out, in, 0, 2, shape)];
                auto d10 = data[pos(out, in, 1, 0, shape)];
                auto d11 = data[pos(out, in, 1, 1, shape)];
                auto d12 = data[pos(out, in, 1, 2, shape)];
                auto d20 = data[pos(out, in, 2, 0, shape)];
                auto d21 = data[pos(out, in, 2, 1, shape)];
                auto d22 = data[pos(out, in, 2, 2, shape)];
                fprintf(stderr, "%5.3f %5.3f %5.3f\n%5.3f %5.3f %5.3f\n%5.3f %5.3f %5.3f", d00, d01, d02, d10, d11, d12, d20, d21, d22);
                cerr << endl;
            }
        }
    };

    // se_type
    std::transform(se_type.begin(), se_type.end(), se_type.begin(), ::tolower);

    // First Block
    //print_conv3x3("input_conv_3x3_w");
    n2n_conv("input_conv_3x3_w", "", false, true);// n2n_input = false, n2n_output = true
    n2n_1D("input_conv_3x3_b", false);

    // ResTower
    for (auto block = 0; block < old_blocks; ++block)
    {
        // ResBlock
        auto prefix = "res_" + to_string(block + 1);
        for (auto layer = 1; layer <= 2; ++layer)
        {
            auto l = to_string(layer);
            auto conv_prefix = prefix + "_conv" + l;
            auto bn_prefix = prefix + "_bn" + l;

            // bn: ((w - m_mean) / m_var) * gamma + beta
            n2n_bn(bn_prefix);

            // conv: w * x + b
            auto remap_type = string{};
            if (layer == 1)
                remap_type = "random";// First convolution in residual block can be widened randomly
            else
                remap_type = "";

            n2n_conv(conv_prefix + "_3x3_w", remap_type, true, true);// n2n_input = true, n2n_output = true
            n2n_1D(conv_prefix + "_3x3_b", false);
        }

        // SE
        auto se_prefix = prefix + "_" + se_type;
        if (se_type == "kt" || se_type == "kt.b")
        {
            // BN->AC->GPool*2->Concat->Conv1x1->Add
            auto conv_prefix = se_prefix + "_conv";
            auto bn_prefix = se_prefix + "_bn";

            // bn: ((w - m_mean) / m_var) * gamma + beta
            n2n_bn(bn_prefix);

            // conv: w * x + b
            n2n_conv(conv_prefix + "_1x1_w", "", true, true, true);// n2n_input = true, n2n_output = true
            n2n_1D(conv_prefix + "_1x1_b", false);
        }
    }

    // Policy Head & Board Value Head & State
    n2n_bn("policy_bn");
    n2n_conv("policy_conv_1x1_w", "", true, false);// n2n_input = true, n2n_output = false
    copy_1D("policy_conv_1x1_b");

    n2n_bn("pass_bn");
    n2n_conv("pass_conv_1x1_w", "", true, false);
    copy_1D("pass_conv_1x1_b");

    n2n_bn("board_value_bn");
    n2n_conv("board_value_conv_1x1_w", "", true, false);
    copy_1D("board_value_conv_1x1_b");

    n2n_bn("state_bn");
    n2n_conv("state_conv_1x1_w", "", true, false);
    copy_1D("state_conv_1x1_b");
}

// ResNetV2 +
// policy_head + board_value_head + state_head +
// se_type in ["", "kt", "kt.b"]
void Network::net2net(uint32_t new_blocks, uint32_t new_filters, std::string se_type)
{
    auto n2n_shape_print = [](std::map<std::string, NDArray>& args_map, std::map<std::string, NDArray>& aux_map, const string& prefix)
    {
        auto print_device_type = [](std::map<std::string, NDArray>& m)
        {
            for (auto& it: m)
            {
                switch (it.second.GetContext().GetDeviceType())
                {
                    case 1: cerr << "CPU -> "; break;
                    case 2: cerr << "GPU -> "; break;
                    case 3: cerr << "CPUPinned -> "; break;
                }
                cerr << it.first << "(";
                for (auto d: it.second.GetShape())
                    cerr << d << ", ";
                cerr << ")" << endl;
            }
        };

        cerr << endl << prefix + " args:" << endl;
        print_device_type(args_map);

        cerr << endl << prefix + " aux:" << endl;
        print_device_type(aux_map);
    };

    // get blocks and filters
    auto n2n_weight_info = [](std::map<std::string, NDArray>& args_map, uint32_t& blocks, uint32_t& filters)
    {
        filters = args_map["input_conv_3x3_b"].GetShape()[0];
        for (blocks = 0; blocks < 1000; ++blocks)
        {
            string name = "res_" + to_string(blocks + 1) + "_conv1_3x3_b";
            if ( args_map.find(name) == args_map.end() )
                break;
        }
        cerr << blocks << "b x " << filters << "f" << endl;
    };

    auto noise = 5e-3f;
    auto dir_alpha = 10.0f;

    std::map<string, NDArray> args_map, aux_map;
    uint32_t old_blocks, old_filters;
    n2n_weight_load(args_map, aux_map);
    n2n_weight_info(args_map, old_blocks, old_filters);
    n2n_shape_print(args_map, aux_map, "old");
    // Leela n2n: 1 deeper, 2 wider.
    n2n_weight_wider(args_map, aux_map, old_blocks, old_filters, new_filters, se_type, noise, dir_alpha);
    n2n_weight_deeper(args_map, aux_map, old_blocks, new_blocks, new_filters, se_type, noise);
    n2n_shape_print(args_map, aux_map, "new");
    n2n_weight_save(args_map, aux_map);
}

void Network::print_conv1x1()
{
    std::map<string, NDArray> args_map, aux_map;
    weight_load(args_map, aux_map);
    for (auto& it: args_map)
    {
        if ( it.first.find("1x1_w") == it.first.npos )
            continue;

        // Get Weight and Bias
        std::string conv_w = it.first;
        std::string conv_b = conv_w;
        conv_b[conv_b.size() - 1] = 'b';

        vector<mx_float> data_w, data_b;
        auto nd_w = args_map[conv_w];
        auto nd_b = args_map[conv_b];
        nd_w.SyncCopyToCPU(&data_w);
        nd_b.SyncCopyToCPU(&data_b);

        auto out_channels = nd_b.Size();
        auto in_channels = data_w.size() / out_channels;

        // print head
        cerr << conv_w << "," << out_channels << "x" << in_channels << ",bias,act,";
        for (size_t c = 0; c < in_channels; ++c)
            cerr << c << ",";
        cerr << endl;

        // print data
        for (size_t out = 0; out < out_channels; ++out)
        {
            auto act_count = 0;
            for (size_t in = 0; in < in_channels; ++in)
            {
                auto v = abs(data_w[in + out * in_channels]);
                if (v > 1e-10)
                    ++act_count;
            }

            cerr << "," << in_channels << "," << data_b[out] << "," << act_count << " / " << in_channels << ",";// bias
            for (size_t in = 0; in < in_channels; ++in)
                cerr << data_w[in + out * in_channels] << ",";// weight
            cerr << endl;
        }
        cerr << endl;
    }
}
