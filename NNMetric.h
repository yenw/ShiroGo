//
// Created by yuanyu on 2020.02.19.
//

#ifndef SHIRO_NNMETRIC_H
#define SHIRO_NNMETRIC_H

#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

class BV_MSE : public EvalMetric
{
public:
    BV_MSE() : EvalMetric("bv_mse") {}
    void Update(NDArray labels, NDArray preds) override {}
    float UpdateMetric(NDArray bv_label, NDArray bv, uint32_t batch_size, uint32_t img_size);
};

class BV_CE : public EvalMetric
{
public:
    BV_CE() : EvalMetric("bv_ce") {}
    void Update(NDArray labels, NDArray preds) override {}
    float UpdateMetric(NDArray bv_label, NDArray bv, uint32_t batch_size, uint32_t img_size);
};

class PI_MSE : public EvalMetric
{
public:
    PI_MSE() : EvalMetric("pi_mse") {}
    void Update(NDArray labels, NDArray preds) override {}
    float UpdateMetric(NDArray policy_label, NDArray policy, uint32_t batch_size);
};

class PI_CE : public EvalMetric
{
public:
    PI_CE() : EvalMetric("pi_ce") {}
    void Update(NDArray labels, NDArray preds) override {}
    float UpdateMetric(NDArray policy_label, NDArray policy, uint32_t batch_size);
};

class S_MSE : public EvalMetric
{
public:
    S_MSE() : EvalMetric("state_mse") {}
    void Update(NDArray labels, NDArray preds) override {}
    float UpdateMetric(NDArray state_label, NDArray state, uint32_t batch_size, uint32_t img_size);
};

class S_CE : public EvalMetric
{
public:
    S_CE() : EvalMetric("state_ce") {}
    void Update(NDArray labels, NDArray preds) override {}
    float UpdateMetric(NDArray state_label, NDArray state, uint32_t batch_size, uint32_t img_size);
};

class NN_LOSS : public EvalMetric
{
public:
    NN_LOSS() : EvalMetric("nn_loss") {}
    void Update(NDArray labels, NDArray preds) override {}
    float UpdateMetric(NDArray loss, uint32_t batch_size);
};

#endif //SHIRO_NNMETRIC_H
