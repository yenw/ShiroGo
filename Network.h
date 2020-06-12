//
// Created by yuanyu on 2018.01.17.
//

#pragma once
#include "mxnet-cpp/MxNetCpp.h"
#include "ReplayPool.h"
#include "Utils.h"

using namespace mxnet::cpp;

class Network
{
public:
    Network(uint32_t width, uint32_t height);
    ~Network();
    void train(ReplayPool& pool, uint32_t iters, uint32_t mini_batch = 0);
    void predict(vector<float>& board_batch, vector<float>& policy_batch, vector<float>& board_value_batch);
    void stopwatch_output(){ sw_predict.output_count("predict time"); }
    void net2net(uint32_t new_blocks, uint32_t new_filters, std::string se_type);
    void print_conv1x1();
    bool is_empty_model(){ return predict_rand; }

private:
    Symbol GetSymbol(uint32_t tower_size, uint32_t num_filter, bool is_train = true);
    void reset_ctx(std::map<std::string, NDArray>& args_map);
    void weight_init(std::map<std::string, NDArray>& args_map, std::map<std::string, NDArray>& aux_map);
    void weight_load(std::map<std::string, NDArray>& args_map, std::map<std::string, NDArray>& aux_map);
    void weight_save(Executor* exec);

    void config_save();
    void config_load();
    
    void arg_map_init(std::map<std::string, NDArray>& args_map, size_t init_batch_size);
    bool is_variable(const std::string& name);

    void n2n_weight_wider(std::map<std::string, NDArray>& args_map, std::map<std::string, NDArray>& aux_map,
                          uint32_t old_blocks, uint32_t old_filters, uint32_t new_filters, std::string se_type,
                          float noise, float dir_alpha);
    void n2n_weight_deeper(std::map<std::string, NDArray>& args_map, std::map<std::string, NDArray>& aux_map,
                           uint32_t old_blocks, uint32_t new_blocks, uint32_t new_filters, std::string se_type,
                           float noise);

    void n2n_weight_save(const std::map<std::string, NDArray>& args_map, const std::map<std::string, NDArray>& aux_map);
    void n2n_weight_load(std::map<std::string, NDArray>& args_map, std::map<std::string, NDArray>& aux_map);

    std::vector<float> dirichlet_dist(float alpha, int size);

private:
    uint32_t img_width, img_height, img_size;
    string folder_model;
    uint32_t train_iter;
    uint32_t body_filters, body_blocks;
    uint32_t batch_size, predict_batch_size;
    uint32_t input_planes;
    Context ctx = Context::cpu();
    int num_gpu;
    bool predict_rand;
    Executor* predict_exec;
    Optimizer* train_opt;
    StopWatch sw_predict, sw_batch;
};

