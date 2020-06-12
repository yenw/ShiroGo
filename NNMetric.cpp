//
// Created by yuanyu on 2020.02.19.
//

#include "NNMetric.h"

float BV_MSE::UpdateMetric(NDArray bv_label, NDArray bv, uint32_t batch_size, uint32_t img_size)
{
    std::vector<mx_float> bv_data;// [-1, 1]
    bv.SyncCopyToCPU(&bv_data);
    std::vector<mx_float> bv_label_data;// [-1, 1]
    bv_label.SyncCopyToCPU(&bv_label_data);

    size_t len = bv_label.Size();
    mx_float sum_loss = 0;
    for (size_t i = 0; i < len; ++i)
    {
        auto diff = bv_label_data[i] - bv_data[i];
        sum_loss += diff * diff * 0.25f;
    }

    sum_loss = sum_loss / float(img_size);
    sum_metric += sum_loss;
    num_inst += batch_size;
    return sum_loss;
}

float BV_CE::UpdateMetric(NDArray bv_label, NDArray bv, uint32_t batch_size, uint32_t img_size)
{
    std::vector<mx_float> bv_data;
    bv.SyncCopyToCPU(&bv_data);
    std::vector<mx_float> bv_label_data;
    bv_label.SyncCopyToCPU(&bv_label_data);

    size_t len = bv_label.Size();
    mx_float sum_loss = 0;
    for (size_t i = 0; i < len; ++i)
        sum_loss += -((bv_label_data[i] + 1.0f) * 0.5f) * log(((bv_data[i] + 1.0f) * 0.5f) + 1e-8f);

    sum_loss = sum_loss / float(img_size);
    sum_metric += sum_loss;
    num_inst += batch_size;
    return sum_loss;
}

float PI_MSE::UpdateMetric(NDArray policy_label, NDArray policy, uint32_t batch_size)
{
    std::vector<mx_float> policy_data;
    policy.SyncCopyToCPU(&policy_data);
    std::vector<mx_float> policy_label_data;
    policy_label.SyncCopyToCPU(&policy_label_data);

    size_t len = policy_label.Size();
    mx_float sum_loss = 0.0f;
    for (size_t i = 0; i < len; ++i)
    {
        auto diff = policy_label_data[i] - policy_data[i];
        sum_loss += diff * diff;
    }

    sum_metric += sum_loss;
    num_inst += batch_size;
    return sum_loss;
}

float PI_CE::UpdateMetric(NDArray policy_label, NDArray policy, uint32_t batch_size)
{
    std::vector<mx_float> policy_data;
    policy.SyncCopyToCPU(&policy_data);
    std::vector<mx_float> policy_label_data;
    policy_label.SyncCopyToCPU(&policy_label_data);

    size_t len = policy_label.Size();
    mx_float sum_loss = 0.0f;
    for (size_t i = 0; i < len; ++i)
        sum_loss += -policy_label_data[i] * log(policy_data[i] + 1e-8f);

    sum_metric += sum_loss;
    num_inst += batch_size;
    return sum_loss;
}

float S_MSE::UpdateMetric(NDArray state_label, NDArray state, uint32_t batch_size, uint32_t img_size)
{
    std::vector<mx_float> state_data;// [0, 1]
    state.SyncCopyToCPU(&state_data);
    std::vector<mx_float> state_label_data;// [0, 1]
    state_label.SyncCopyToCPU(&state_label_data);

    size_t len = state_label.Size();
    mx_float sum_loss = 0;
    for (size_t i = 0; i < len; ++i)
    {
        auto diff = state_label_data[i] - state_data[i];
        sum_loss += diff * diff;
    }

    sum_loss = sum_loss / float(img_size * 3);
    sum_metric += sum_loss;
    num_inst += batch_size;
    return sum_loss;
}

float S_CE::UpdateMetric(NDArray state_label, NDArray state, uint32_t batch_size, uint32_t img_size)
{
    std::vector<mx_float> state_data;// [0, 1]
    state.SyncCopyToCPU(&state_data);
    std::vector<mx_float> state_label_data;// [0, 1]
    state_label.SyncCopyToCPU(&state_label_data);

    size_t len = state_label.Size();
    mx_float sum_loss = 0;
    for (size_t i = 0; i < len; ++i)
        sum_loss += -state_label_data[i] * log(state_data[i] + 1e-8f);

    sum_loss = sum_loss / float(img_size * 3);
    sum_metric += sum_loss;
    num_inst += batch_size;
    return sum_loss;
}

float NN_LOSS::UpdateMetric(NDArray loss, uint32_t batch_size)
{
    std::vector<mx_float> loss_data;
    loss.SyncCopyToCPU(&loss_data);

    size_t len = loss.Size();
    mx_float batch_loss = 0.0f;
    for (size_t i = 0; i < len; ++i)
        batch_loss += loss_data[i];

    sum_metric += batch_loss;
    num_inst += batch_size;
    return batch_loss;
}