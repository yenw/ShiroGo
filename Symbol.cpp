//
// Created by yuanyu on 2018.05.26.
//

#include <algorithm>
#include "Symbol.h"
#include "config.h"

auto stride_1x1 = Shape(1, 1);
auto stride_2x2 = Shape(2, 2);
auto dilate_1x1 = Shape(1, 1);
auto dilate_2x2 = Shape(2, 2);
auto group_1 = 1;

auto split = [](std::string& s, std::string delim)
{
    size_t last = 0;
    size_t index = s.find_first_of(delim, last);
    std::vector<std::string> ret;
    while (index != std::string::npos)
    {
        ret.push_back(s.substr(last, index - last));
        last = index + 1;
        index = s.find_first_of(delim, last);
    }

    if (index - last > 0)
        ret.push_back(s.substr(last, index - last));

    return ret;
};

// Basic
Symbol Conv(const std::string& name, Symbol data, Shape kernel, uint32_t num_filter, uint32_t num_group,
            Shape stride, Shape dilate, Shape pad)
{
    auto conv_name = name + "_" + std::to_string(kernel[0]) + "x" + std::to_string(kernel[1]);
    auto conv_w = Symbol(conv_name + "_w");
    auto conv_b = Symbol(conv_name + "_b");
    auto conv = Convolution(conv_name, data, conv_w, conv_b, kernel, num_filter, stride, dilate, pad, num_group,
                1024, false, ConvolutionCudnnTune::kFastest);
    return conv;
}

Symbol IterConv(const std::string& name, Symbol data, Shape kernel, uint32_t iter, uint32_t num_filter, uint32_t num_group,
                Shape stride, Shape dilate, Shape pad)
{
    auto conv_name = name + "_" + std::to_string(kernel[0]) + "x" + std::to_string(kernel[1]);
    auto conv_w = Symbol(conv_name + "_w");
    auto conv_b = Symbol(conv_name + "_b");

    for (uint32_t i = 0; i < iter; ++i)
    {
        auto iter_name = conv_name + "_" + std::to_string(i + 1);
        data = Convolution(iter_name, data, conv_w, conv_b, kernel, num_filter, stride, dilate, pad, num_group,
                           1024, false, ConvolutionCudnnTune::kFastest);
    }

    return data;
}

Symbol Conv3x3(const std::string& name, Symbol data, uint32_t num_filter, uint32_t num_group,
               Shape stride, Shape dilate, Shape pad)
{
    return Conv(name, data, Shape(3, 3), num_filter, num_group, stride, dilate, pad);
}

Symbol Conv1x1(const std::string& name, Symbol data, uint32_t num_filter, uint32_t num_group,
               Shape stride, Shape dilate, Shape pad)
{
    return Conv(name, data, Shape(1, 1), num_filter, num_group, stride, dilate, pad);
}

Symbol Conv1x3(const std::string& name, Symbol data, uint32_t num_filter, uint32_t num_group,
               Shape stride, Shape dilate, Shape pad)
{
    return Conv(name, data, Shape(1, 3), num_filter, num_group, stride, dilate, pad);
}

Symbol Conv3x1(const std::string& name, Symbol data, uint32_t num_filter, uint32_t num_group,
               Shape stride, Shape dilate, Shape pad)
{
    return Conv(name, data, Shape(3, 1), num_filter, num_group, stride, dilate, pad);
}

Symbol Conv1x5(const std::string& name, Symbol data, uint32_t num_filter, uint32_t num_group,
               Shape stride, Shape dilate, Shape pad)
{
    return Conv(name, data, Shape(1, 5), num_filter, num_group, stride, dilate, pad);
}

Symbol Conv5x1(const std::string& name, Symbol data, uint32_t num_filter, uint32_t num_group,
               Shape stride, Shape dilate, Shape pad)
{
    return Conv(name, data, Shape(5, 1), num_filter, num_group, stride, dilate, pad);
}

Symbol Conv1x7(const std::string& name, Symbol data, uint32_t num_filter, uint32_t num_group,
               Shape stride, Shape dilate, Shape pad)
{
    return Conv(name, data, Shape(1, 7), num_filter, num_group, stride, dilate, pad);
}

Symbol Conv7x1(const std::string& name, Symbol data, uint32_t num_filter, uint32_t num_group,
               Shape stride, Shape dilate, Shape pad)
{
    return Conv(name, data, Shape(7, 1), num_filter, num_group, stride, dilate, pad);
}

Symbol IterConv3x3(const std::string& name, Symbol data, uint32_t iter, uint32_t num_filter, uint32_t num_group,
                   Shape stride, Shape dilate, Shape pad)
{
    return IterConv(name, data, Shape(3, 3), iter, num_filter, num_group, stride, dilate, pad);
}

Symbol GN(const std::string& name, Symbol data, int num_groups, float eps)
{
    auto gamma = Symbol(name + "_gamma");
    auto beta = Symbol(name + "_beta");
    auto gn = GroupNorm(name, data, gamma, beta, num_groups, eps);
    return gn;
}

Symbol LN(const std::string& name, Symbol data, int axis, float eps)
{
    auto gamma = Symbol(name + "_gamma");
    auto beta = Symbol(name + "_beta");
    auto ln = LayerNorm(name, data, gamma, beta, axis, eps);
    return ln;
}

Symbol IN(const std::string& name, Symbol data, float eps)
{
    auto gamma = Symbol(name + "_gamma");
    auto beta = Symbol(name + "_beta");
    auto in = InstanceNorm(name, data, gamma, beta, eps);
    return in;
}

Symbol BN(const std::string& name, Symbol data, double eps, mx_float momentum, bool fix_gamma, bool use_global_stats)
{
    auto gamma = Symbol(name + "_gamma");
    auto beta = Symbol(name + "_beta");
    auto moving_mean = Symbol(name + "_moving_mean");
    auto moving_var = Symbol(name + "_moving_var");
    auto bn = BatchNorm(name, data, gamma, beta, moving_mean, moving_var, eps, momentum, fix_gamma, use_global_stats);
    return bn;
}

Symbol Norm(const std::string& name, Symbol data, std::string norm_type, std::string idx)
{
    std::transform(norm_type.begin(), norm_type.end(), norm_type.begin(), ::tolower);
    if (norm_type == "gn")
        return GN(name + "_gn" + idx, data);
    else if (norm_type == "ln")
        return LN(name + "_ln" + idx, data);
    else if (norm_type == "in")
        return IN(name + "_in" + idx, data);
    else if (norm_type == "bn")
        return BN(name + "_bn" + idx, data);
    else
        return data;
}

Symbol AC(const std::string& name, Symbol data, std::string act_type, std::string idx)
{
    if (act_type == "")
        act_type = "relu";

    std::transform(act_type.begin(), act_type.end(), act_type.begin(), ::tolower);
    if (act_type == "relu" || act_type == "sigmoid" || act_type == "softrelu" || act_type == "softsign" || act_type == "tanh")
        return Activation(name + "_" + act_type + idx, data, act_type);
    else// elu, leaky, rrelu, prelu
    {
        Symbol sym;
        auto type = LeakyReLUActType::kElu;
        auto slope = 0.25f;
        auto lower_bound = 0.125f;
        auto upper_bound = 0.334f;
        auto param = split(act_type, "_");
        if (param.size() > 0)
        {
            auto lrelu = param[0];
            if (lrelu == "elu")
            {
                type = LeakyReLUActType::kElu;
                if (param.size() > 1)
                    slope = stof(param[1]);
            }
            else if (lrelu == "gelu")
            {
                type = LeakyReLUActType::kGelu;
            }
#ifdef MXNET_PR
            else if (lrelu == "mish")
            {
                type = LeakyReLUActType::kMish;
            }
            else if (lrelu == "swish1")
            {
                type = LeakyReLUActType::kSwish1;
            }
#endif
            else if (lrelu == "leaky")
            {
                type = LeakyReLUActType::kLeaky;
                if (param.size() > 1)
                    slope = stof(param[1]);
            }
            else if (lrelu == "prelu")
            {
                type = LeakyReLUActType::kPrelu;
                sym = Symbol(name + "_gamma");
            }
            else if (lrelu == "rrelu")
            {
                type = LeakyReLUActType::kRrelu;
                if (param.size() > 2)
                {
                    lower_bound = stof(param[1]);
                    upper_bound = stof(param[2]);
                }
            }
            else if (lrelu == "selu")
            {
                // 学习率不能太高, 否则损失降不下来.
                type = LeakyReLUActType::kSelu;// mxnet.version >= 1.3.1
            }
        }

        return LeakyReLU(name+ "_" + act_type + idx, data, sym, type, slope, lower_bound, upper_bound);
    }
}

Symbol FC(const std::string& name, Symbol data, int num_hidden, bool no_bias, bool flatten)
{
    auto fc_w = Symbol(name + "_w");
    auto fc_b = Symbol(name + "_b");
    return FullyConnected(name, data, fc_w, fc_b, num_hidden, no_bias, flatten);
}

// Block
Symbol Norm_AC(const std::string& name, Symbol data, std::string act_type, std::string norm_type, std::string idx)
{
    if (act_type != "selu")
    {
        auto norm = Norm(name, data, norm_type, idx);
        auto ac = AC(name, norm, act_type, idx);
        return ac;
    }
    else
        return AC(name, data, act_type, idx);
}

Symbol Conv3x3_Norm_AC(const std::string& name, Symbol data, uint32_t num_filter, std::string act_type, std::string norm_type,
                       std::string idx, uint32_t num_group, Shape stride)
{
    auto conv = Conv3x3(name + "_conv" + idx, data, num_filter, num_group, stride);
    return Norm_AC(name, conv, act_type, norm_type, idx);
}

Symbol Norm_AC_Conv3x3(const std::string& name, Symbol data, uint32_t num_filter, std::string act_type, std::string norm_type,
                       std::string idx, uint32_t num_group)
{
    auto norm_ac = Norm_AC(name, data, act_type, norm_type, idx);
    return Conv3x3(name + "_conv" + idx, norm_ac, num_filter, num_group);
}

Symbol Norm_AC_Conv1x1(const std::string& name, Symbol data, uint32_t num_filter, std::string act_type, std::string norm_type,
                       std::string idx, uint32_t num_group)
{
    auto norm_ac = Norm_AC(name, data, act_type, norm_type, idx);
    return Conv1x1(name + "_conv" + idx, norm_ac, num_filter, num_group);
}

Symbol SE_Block(const std::string& name, Symbol data, uint32_t num_filter)
{
    // squeeze & excitation
    auto avg_pool = Pooling(name + "_avg_pool", data, Shape(), PoolingPoolType::kAvg, true);
    auto fc1 = Conv1x1(name + "_conv1", avg_pool, num_filter / 16);
    auto ac1 = AC(name, fc1, "relu");
    auto fc2 = Conv1x1(name + "_conv2", ac1, num_filter);
    auto ac2 = AC(name, fc2, "sigmoid");
    auto se = broadcast_mul(name + "_mul", data, ac2);

    return se;
}

Symbol SNSE_Block(const std::string& name, Symbol data, uint32_t num_filter, std::string norm_type)
{
    // SparseNet: squeeze & excitation
    auto avg_pool = Pooling(name + "_avg_pool", data, Shape(), PoolingPoolType::kAvg, true);
    auto conv1 = Norm_AC_Conv1x1(name, avg_pool, num_filter, "relu", norm_type,"1");
    auto res = conv1 + avg_pool;

    auto conv2 = Norm_AC_Conv1x1(name, res, num_filter, "relu", norm_type, "2");
    auto snse = broadcast_mul(name + "_mul", data, conv2);

    return snse;
}

Symbol SA_Block(const std::string& name, Symbol data)
{
    // CBAM: Spatial Attention
    auto max_pool = max(name + "_max", data, dmlc::optional<Shape>(Shape(1)), true);// [batch, channel, height, width] --> [batch, 1, height, width]
    auto avg_pool = mean(name + "_mean", data, dmlc::optional<Shape>(Shape(1)), true);
    auto concat = Concat(name + "_concat", {max_pool, avg_pool}, 2);
    auto conv = Conv1x1(name + "_conv", concat, 1);
    auto ac = AC(name, conv, "sigmoid");
    auto sa = broadcast_mul(name + "_mul", data, ac);

    return sa;
}

Symbol CBAM_Block(const std::string& name, Symbol data, uint32_t num_filter)
{
    // CBAM: Channel attention
    auto ca = [](std::string name, Symbol data, uint32_t num_filter, PoolingPoolType pool_type)
    {
        auto pool = Pooling(name + "_pool", data, Shape(), pool_type, true);
        auto fc1 = Conv1x1(name + "_conv1", pool, num_filter / 2);
        auto ac1 = AC(name, fc1, "relu");
        auto fc2 = Conv1x1(name + "_conv2", ac1, num_filter);
        auto ac2 = AC(name, fc2, "sigmoid");
        return ac2;
    };
    auto avg_ca = ca(name + "_se", data, num_filter, PoolingPoolType::kAvg);
    auto max_ca = ca(name + "_max", data, num_filter, PoolingPoolType::kMax);
    data = broadcast_mul(name + "_mul", data, avg_ca + max_ca);

    // CBAM: Spatial Attention
    auto sa_name = name + "_sa";
    auto max_pool = max(sa_name + "_max", data, dmlc::optional<Shape>(Shape(1)), true);// [batch, channel, height, width] --> [batch, 1, height, width]
    auto avg_pool = mean(sa_name + "_mean", data, dmlc::optional<Shape>(Shape(1)), true);
    auto concat = Concat(sa_name + "_concat", {max_pool, avg_pool}, 2);
    auto conv = Conv1x1(sa_name + "_conv", concat, 1);
    auto ac = AC(sa_name, conv, "sigmoid");
    data = broadcast_mul(sa_name + "_mul", data, ac);

    return data;
}

Symbol SAA_Block(const std::string& name, Symbol data, uint32_t num_filter, std::string act_type, std::string norm_type)
{
    // Spatial Attention & Add
    auto norm_ac = Norm_AC(name, data, act_type, norm_type);
    auto max_pool = Pooling(name + "_global_avgpool", norm_ac, Shape(), PoolingPoolType::kMax, true);// [batch, channel, height, width] --> [batch, C, 1, 1]
    auto avg_pool = Pooling(name + "_global_maxpool", norm_ac, Shape(), PoolingPoolType::kAvg, true);
    auto concat = Concat(name + "_concat", {max_pool, avg_pool}, 2);
    auto fc1 = Conv1x1(name + "_conv1", concat, num_filter);
    // LayerNorm(fc1)
    auto ac1 = AC(name, fc1, act_type);
    auto fc2 = Conv1x1(name + "_conv2", ac1, num_filter);
    data = broadcast_add(name + "_add", data, fc2);

    return data;
}

Symbol KT_Block(const std::string& name, Symbol data, uint32_t num_filter, std::string act_type, std::string norm_type)
{
    // KataGo Block
    auto norm_ac = Norm_AC(name, data, act_type, norm_type);
    auto max_pool = Pooling(name + "_global_avgpool", norm_ac, Shape(), PoolingPoolType::kMax, true);// [batch, channel, height, width] --> [batch, C, 1, 1]
    auto avg_pool = Pooling(name + "_global_maxpool", norm_ac, Shape(), PoolingPoolType::kAvg, true);
    auto concat = Concat(name + "_concat", {max_pool, avg_pool}, 2);
    auto fc1 = Conv1x1(name + "_conv", concat, num_filter);
    data = broadcast_add(name + "_add", data, fc1);

    return data;
}

Symbol SE(const std::string& name, Symbol data, uint32_t num_filter,
          std::string act_type, std::string norm_type, std::string se_type, bool is_se_type_b)
{
    std::string suffix = is_se_type_b ? ".b" : "";
    std::string suffix_upper = is_se_type_b ? ".B" : "";
    for (auto se: split(se_type, "_"))
    {
        std::transform(se.begin(), se.end(), se.begin(), ::toupper);
        if (se == ("SE" + suffix_upper))
            data = SE_Block(name + "_se" + suffix, data, num_filter);
        else if (se == ("SNSE" + suffix_upper))
            data = SNSE_Block(name + "_snse" + suffix, data, num_filter, norm_type);
        else if (se == ("SA" + suffix_upper))
            data = SA_Block(name + "_sa" + suffix, data);
        else if (se == ("CBAM" + suffix_upper))
            data = CBAM_Block(name + "_cbam" + suffix, data, num_filter);
        else if (se == ("SAA" + suffix_upper))
            data = SAA_Block(name + "_saa" + suffix, data, num_filter, act_type, norm_type);
        else if (se == ("KT" + suffix_upper))
            data = KT_Block(name + "_kt" + suffix, data, num_filter, act_type, norm_type);
    }
    // 未实现
    // Residual Attention Network for Image Classification

    return data;
}

Symbol ResBlock(const std::string& name, Symbol data, uint32_t num_filter, bool bottle_neck,
                std::string act_type, std::string norm_type, std::string se_type)
{
    if (bottle_neck)
    {
        auto conv1 = Conv1x1(name + "_conv1", data, num_filter / 2);
        auto norm1 = Norm(name, conv1, norm_type, "1");
        auto ac1 = AC(name, norm1, act_type, "1");

        auto conv2 = Conv3x3(name + "_conv2", ac1, num_filter / 2);
        auto norm2 = Norm(name, conv2, norm_type, "2");
        auto ac2 = AC(name, norm2, act_type, "2");

        auto conv3 = Conv1x1(name + "_conv3", ac2, num_filter);
        auto norm3 = Norm(name, conv3, norm_type, "3");

        auto fused = data + norm3;
        return AC(name, fused, act_type, "3");
    }
    else
    {
        auto conv1 = Conv3x3(name + "_conv1", data, num_filter);
        auto norm1 = Norm(name, conv1, norm_type, "1");
        auto ac1 = AC(name, norm1, act_type, "1");

        auto conv2 = Conv3x3(name + "_conv2", ac1, num_filter);
        auto norm2 = Norm(name, conv2, norm_type, "2");

        auto fused = data + norm2;
        return AC(name, fused, act_type, "2");
    }
}

Symbol ResV2Block(const std::string& name, Symbol data, uint32_t num_filter, uint32_t blocks, bool bottle_neck,
                  std::string act_type, std::string norm_type, std::string se_type)
{
    if (bottle_neck)
    {
        --blocks;
        data = Norm_AC_Conv1x1(name, data, num_filter / 2, act_type, norm_type,"1");
        for (uint32_t i = 2; i <= blocks + 1; ++i)
            data = Norm_AC_Conv3x3(name, data, num_filter / 2, act_type, norm_type, std::to_string(i));

        data = Norm_AC_Conv1x1(name, data, num_filter, act_type, norm_type, std::to_string(blocks + 2));
        return SE(name, data, num_filter, act_type, norm_type, se_type);
    }
    else
    {
        for (uint32_t i = 1; i <= blocks; ++i)
            data = Norm_AC_Conv3x3(name, data, num_filter, act_type, norm_type, std::to_string(i));

        return SE(name, data, num_filter, act_type, norm_type, se_type);
    }
}

Symbol ResNeXtBlock(const std::string& name, Symbol data, uint32_t num_filter, bool bottle_neck,
                    std::string act_type, std::string norm_type, std::string se_type)
{
    if (bottle_neck)
    {
        auto num_group = std::max(uint32_t{1}, std::min(uint32_t{32}, num_filter / 8));
        auto conv1 = Norm_AC_Conv1x1(name, data, num_filter / 2, act_type, norm_type,"1");
        auto conv2 = Norm_AC_Conv3x3(name, conv1, num_filter / 2, act_type, norm_type, "2", num_group);
        auto conv3 = Norm_AC_Conv1x1(name, conv2, num_filter, act_type, norm_type, "3");

        return SE(name, conv3, num_filter, act_type, norm_type, se_type);
    }
    else
    {
        auto conv1 = Norm_AC_Conv3x3(name, data, num_filter, act_type, norm_type, "1");
        auto conv2 = Norm_AC_Conv3x3(name, conv1, num_filter, act_type, norm_type, "2");

        return SE(name, conv2, num_filter, act_type, norm_type, se_type);
    }
}

// Tower
Symbol ResNetTower(const std::string& name, Symbol data, uint32_t num_filter, uint32_t tower_size, bool bottle_neck,
                   std::string act_type, std::string norm_type, std::string se_type)
{
    for (uint32_t i = 1; i <= tower_size; ++i)
    {
        data = ResBlock(name + "_" + std::to_string(i), data, num_filter, bottle_neck, act_type, norm_type, se_type);
        data = SE(name + "_" + std::to_string(i), data, num_filter, act_type, norm_type, se_type, true);
    }

    return data;
}

Symbol ResNetV2Tower(const std::string& name, Symbol data, uint32_t num_filter, uint32_t tower_size,
                     bool bottle_neck, std::string act_type, std::string norm_type, std::string se_type)
{
    for (uint32_t i = 1; i <= tower_size; ++i)
    {
        data = data + ResV2Block(name + "_" + std::to_string(i), data, num_filter, 2, bottle_neck, act_type, norm_type, se_type);
        data = SE(name + "_" + std::to_string(i), data, num_filter, act_type, norm_type, se_type, true);
    }

    return data;
}

Symbol ResNeXtTower(const std::string& name, Symbol data, uint32_t num_filter, uint32_t tower_size,
                    bool bottle_neck, std::string act_type, std::string norm_type, std::string se_type)
{
    for (uint32_t i = 1; i <= tower_size; ++i)
    {
        data = data + ResNeXtBlock(name + "_" + std::to_string(i), data, num_filter, bottle_neck, act_type, norm_type, se_type);
        data = SE(name + "_" + std::to_string(i), data, num_filter, act_type, norm_type, se_type, true);
    }

    return data;
}

Symbol ResNetCZTower(const std::string& name, Symbol data, uint32_t num_filter, uint32_t tower_size,
                     bool bottle_neck, std::string act_type, std::string norm_type, std::string se_type)
{
    auto idx = [](uint32_t i)
    {
        auto bit = i & (i - 1);
        if (bit)
            return i - (i ^ bit) * 2;
        else
            return i / 2;
    };

    std::vector<Symbol> layer{data};
    for (uint32_t i = 1; i <= tower_size; ++i)
    {
        if (i > 1)
            data = layer[idx(i + 1) - 1] + ResV2Block(name + "_" + std::to_string(i), data, num_filter, 1, bottle_neck, act_type, norm_type, se_type);
        else
            data = ResV2Block(name + "_" + std::to_string(i), data, num_filter, 1, bottle_neck, act_type, norm_type, se_type);

        data = SE(name + "_" + std::to_string(i), data, num_filter, act_type, norm_type, se_type, true);
        layer.push_back(data);
    }

    return layer.back();
}

// DPN
std::vector<Symbol> DPNInput(const std::string& name, Symbol data, uint32_t num_filter, uint32_t inc)
{
    auto conv = Conv3x3(name + "_conv", data, num_filter + inc);
    auto res = slice_axis(name + "_res", conv, 1, 0, dmlc::optional<int>(num_filter));
    auto dense = slice_axis(name + "_dense", conv, 1, num_filter, dmlc::optional<int>(num_filter + inc));
    return {res, dense};
}

std::vector<Symbol> DPNBlock(const std::string& name, std::vector<Symbol> input, uint32_t num_filter, uint32_t inc,
                             std::string act_type, std::string norm_type, std::string se_type)
{
    auto data = Concat(name + "_input_concat", input, input.size());

    // resnext
    auto res_name = name + "_res";
    auto num_group = std::max(uint32_t{1}, std::min(uint32_t{64}, num_filter / 8));

    auto conv1 = Norm_AC_Conv1x1(res_name, data, num_filter / 2, act_type, norm_type, "1");
    auto conv2 = Norm_AC_Conv3x3(res_name, conv1, num_filter / 2, act_type, norm_type, "2", num_group);
    auto norm_ac3 = Norm_AC(res_name, conv2, act_type, norm_type, "3");
    auto conv3 = Conv1x1(res_name + "_conv3", norm_ac3, num_filter);
    auto fused = input[0] + SE(res_name, conv3, num_filter, act_type, norm_type, se_type);// SE before
    fused = SE(res_name, fused, num_filter, act_type, norm_type, se_type, true);// SE after

    // densenet
    auto dense_name = name + "_dense";
    auto dense_conv = Conv1x1(dense_name + "_conv", norm_ac3, inc);
    std::vector<Symbol> dense{input[1], dense_conv};

    std::vector<Symbol> output;
    output.push_back(fused);
    output.push_back(Concat(name + "_output_concat", dense, dense.size()));
    return output;
}

Symbol DPNTower(const std::string& name, std::vector<Symbol> input, uint32_t num_filter, uint32_t tower_size, uint32_t inc,
                std::string act_type, std::string norm_type, std::string se_type)
{
    for (uint32_t i = 1; i <= tower_size; ++i)
        input = DPNBlock(name + "_" + std::to_string(i), input, num_filter, inc, act_type, norm_type, se_type);

    auto final = Concat(name + "_final_concat", input, input.size());
    return final;
}

// MixNet
std::vector<Symbol> MixNetInput(const std::string& name, Symbol data, uint32_t inc)
{
    auto res_conv = Conv3x3(name + "_res_conv", data, inc);
    auto dense_conv = Conv3x3(name + "_dense_conv", data, inc);
    return {dense_conv, res_conv};
}

std::vector<Symbol> MixNetBlock(const std::string& name, Symbol data, uint32_t inc, bool sparse,
                                std::string act_type, std::string norm_type, std::string se_type)
{
    if (sparse)
    {
        // res: 3x3
        auto res_name = name + "_res";
        auto res_conv = Norm_AC_Conv3x3(res_name, data, inc, act_type, norm_type);
        auto res_se = SE(res_name, res_conv, inc, act_type, norm_type, se_type);

        // densenet: 3x3
        auto dense_name = name + "_dense";
        auto dense_conv = Norm_AC_Conv3x3(dense_name, data, inc, act_type, norm_type);

        return {dense_conv, res_se};
    }
    else
    {
        // res: 1x1, 3x3
        auto res_name = name + "_res";
        auto res_conv1 = Norm_AC_Conv1x1(res_name, data, inc * 4, act_type, norm_type, "1");
        auto res_conv2 = Norm_AC_Conv3x3(res_name, res_conv1, inc, act_type, norm_type, "2");
        auto res_se = SE(res_name, res_conv2, inc, act_type, norm_type, se_type);

        // densenet: 1x1, 3x3
        auto dense_name = name + "_dense";
        auto dense_conv1 = Norm_AC_Conv1x1(dense_name, data, inc * 4, act_type, norm_type, "1");
        auto dense_conv2 = Norm_AC_Conv3x3(dense_name, dense_conv1, inc, act_type, norm_type, "2");

        return {dense_conv2, res_se};
    }
}

Symbol MixNetTower(const std::string &name, std::vector<Symbol> data, uint32_t inc, uint32_t tower_size, bool sparse,
                   std::string act_type, std::string norm_type, std::string se_type)
{
    auto block_input = [](std::string name, std::vector<Symbol>& layer, bool sparse)
    {
        if (sparse)
        {
            std::vector<Symbol> sparse_layer;
            size_t back = 1;
            size_t layer_size = layer.size();
            while (back <= layer_size)
            {
                // i - 1, i - 2, i - 4, ...
                sparse_layer.push_back(layer[layer_size - back]);
                back *= 2;
            }

            auto concat = Concat(name + "_sparse_concat", sparse_layer, sparse_layer.size());
            return concat;
        }
        else
        {
            auto concat = Concat(name + "_concat", layer, layer.size());
            return concat;
        }
    };

    auto idx = [](size_t i)
    {
        auto bit = i & (i - 1);
        if (bit)
            return i - (i ^ bit) * 2;
        else
            return i / 2;
    };

    std::vector<Symbol> layer{data[0], data[1]};
    auto back = layer.size() - 1;
    for (size_t i = 1; i <= tower_size; ++i, ++back)
    {
        auto layer_name = name + "_" + std::to_string(i);
        auto input = block_input(layer_name, layer, sparse);
        auto output = MixNetBlock(layer_name, input, inc, sparse, act_type, norm_type, se_type);
        auto back_idx = idx(back);
        layer[back_idx] = layer[back_idx] + output[1];// back
        layer[back_idx] = SE(layer_name, layer[back_idx], inc, act_type, norm_type, se_type, true);// SE after
        layer.push_back(output[0]);
    }

    auto final = Concat(name + "_final_concat", layer, layer.size());
    return final;
}

// Iter MixNet
Symbol IterBlock(const std::string& name, Symbol data, uint32_t num_filter, uint32_t iter)
{
    auto init_conv = Conv3x3(name + "_init_conv", data, num_filter);
    auto iter_conv = IterConv3x3(name + "_iter_conv", init_conv, iter, num_filter);
    return iter_conv;
}

std::vector<Symbol> IterMixNetInput(const std::string &name, Symbol data, uint32_t inc, uint32_t iter)
{
    // IterBlock
    auto st = slice_axis(name + "_st", data, 1, 0, dmlc::optional<int>(2));
    auto ib = IterBlock(name + "_ib", st, inc, iter);
    //auto ib = IterBlock(name + "_ib", data, inc, iter);

    // dense & res
    auto res_conv = Conv3x3(name + "_res_conv", data, inc);
    auto dense_conv = Conv3x3(name + "_dense_conv", data, inc);
    return {dense_conv, ib, res_conv};
}

// Policy Head
Symbol PolicyHead(Symbol data, bool norm_ac, std::string act_type, std::string norm_type)
{
    if (norm_ac)
        data = Norm_AC("policy", data, act_type, norm_type);

    return Conv1x1("policy_conv", data, 1);
}

Symbol PassHead(Symbol data, bool norm_ac, std::string act_type, std::string norm_type)
{
    if (norm_ac)
        data = Norm_AC("pass", data, act_type, norm_type);

    return Conv1x1("pass_conv", data, 1);
}

// BoardValue Head
Symbol BoardValueHead(Symbol data, bool norm_ac, std::string act_type, std::string norm_type)
{
    if (norm_ac)
        data = Norm_AC("board_value", data, act_type, norm_type);

    auto bv_conv = Conv1x1("board_value_conv", data, 2);
    #if MXNET_VERSION >= 10600
        return softmax("board_value_softmax", bv_conv, Symbol(), 1);// >= 1.6.0
    #else
        return softmax("board_value_softmax", bv_conv, 1);
    #endif
}

// State Head
Symbol StateHead(Symbol data, bool norm_ac, std::string act_type, std::string norm_type)
{
    if (norm_ac)
        data = Norm_AC("state", data, act_type, norm_type);

    auto state_conv = Conv1x1("state_conv", data, 3);
    #if MXNET_VERSION >= 10600
        return softmax("state_softmax", state_conv, Symbol(), 1);
    #else
        return softmax("state_softmax", state_conv, 1);
    #endif
}

// Loss
Symbol Loss(Symbol policy, Symbol pass, Symbol board_value, Symbol state,
            Symbol policy_mcts, Symbol board_value_mcts, Symbol next_state, Symbol rate, uint32_t img_size)
{
    auto cross_entropy = [&](Symbol data, Symbol label)
    {
        return negative(label * log(data + 1e-8f));
    };

    auto cross_entropy_bw = [&](Symbol black, Symbol white, Symbol label)
    {
        return negative(label * log(black + 1e-8f) + (1.0f - label) * log(white + 1e-8f));
    };

    auto sum_board = [&](Symbol board)
    {
        return sum(board, dmlc::optional<Shape>(Shape(1)), true);//[batch, height * width] --> [batch, 1]
    };

    auto avg_board = [&](Symbol board)
    {
        return mean(board, dmlc::optional<Shape>(Shape(1)), true);//[batch, height * width] --> [batch, 1]
    };

    // policy: (2, height, width) --> height * width + 1
    auto policy_board = Flatten("policy_flatten", policy);// [batch, 1, height, width] --> [batch, height * width]
    auto policy_pass = Flatten("policy_pass", Pooling("policy_pool", pass, Shape(), PoolingPoolType::kAvg, true));// [batch, 1, height, width] --> [batch, 1]
    auto concat_list = std::vector<Symbol>{policy_board, policy_pass};
    auto policy_concat = Concat("policy_concat", concat_list, concat_list.size());
#if MXNET_VERSION >= 10600
    auto policy_output = softmax("policy_softmax", policy_concat, Symbol());
#else
    auto policy_output = softmax("policy_softmax", policy_concat);
#endif

    // board value
    auto bv_white_slice = slice_axis("board_value_white_slice", board_value, 1, 0, dmlc::optional<int>(1));
    auto bv_black_slice = slice_axis("board_value_black_slice", board_value, 1, 1, dmlc::optional<int>(2));
    auto bv_white = Flatten("board_value_white_flatten", bv_white_slice);
    auto bv_black = Flatten("board_value_black_flatten", bv_black_slice);
    auto bv_output = bv_black * 2.0f - 1.0f;

    // value
    auto value_white = avg_board(bv_white);
    auto value_black = avg_board(bv_black);
    auto value_mcts = avg_board(board_value_mcts);
    
    // state
    auto state_output = Flatten("state_flatten", state);

    // loss
    auto policy_loss = sum_board(cross_entropy(policy_output, policy_mcts));
    auto bv_loss = avg_board(cross_entropy_bw(bv_black, bv_white, (1.0f + board_value_mcts) / 2.0f));
    auto value_loss = cross_entropy_bw(value_black, value_white, (1.0f + value_mcts) / 2.0f);
    auto state_loss = avg_board(cross_entropy(state_output, next_state));
    //auto entropy_bonus = sum_board(cross_entropy(policy_output, policy_output));
    //auto entropy_rate = 0.01;

    // final loss
    //auto net_loss = (std::log2(float(img_size)) * 0.5f) * (bv_loss + value_loss) + policy_loss + entropy_rate * entropy_bonus;
    auto v_loss_weight = (std::log2(float(img_size)) * 0.5f) * (bv_loss + value_loss);
    auto s_loss_weight = std::log(float(img_size)) / std::log(3.0f) * state_loss * (ACTION_PLANES > 0);
    auto net_loss = (v_loss_weight + s_loss_weight + policy_loss) * rate;
    auto loss = MakeLoss("Final_Loss", net_loss);
    return Symbol::Group({BlockGrad(policy_output), BlockGrad(bv_output), BlockGrad(state_output), loss});
}

Symbol Head(Symbol tower, Symbol policy_mcts, Symbol board_value_mcts, Symbol next_state, Symbol rate,
            uint32_t img_size, bool norm_ac, std::string act_type, std::string norm_type)
{
    // policy & pass
    auto policy = PolicyHead(tower, norm_ac, act_type, norm_type);
    auto pass = PassHead(tower, norm_ac, act_type, norm_type);

    // board value
    auto board_value = BoardValueHead(tower, norm_ac, act_type, norm_type);
    
    // state
    auto state = StateHead(tower, norm_ac, act_type, norm_type);

    // loss
    return Loss(policy, pass, board_value, state, policy_mcts, board_value_mcts, next_state, rate, img_size);
}

Symbol OutputHead(Symbol tower, bool norm_ac, std::string act_type, std::string norm_type)
{
    // policy & pass
    auto policy = PolicyHead(tower, norm_ac, act_type, norm_type);
    auto pass = PassHead(tower, norm_ac, act_type, norm_type);

    // board value
    auto board_value = BoardValueHead(tower, norm_ac, act_type, norm_type);

    // state
    auto state = StateHead(tower, norm_ac, act_type, norm_type);

    // policy: (2, height, width) --> height * width + 1
    auto policy_board = Flatten("policy_flatten", policy);// [batch, 1, height, width] --> [batch, height * width]
    auto policy_pass = Flatten("policy_pass", Pooling("policy_pool", pass, Shape(), PoolingPoolType::kAvg, true));// [batch, 1, height, width] --> [batch, 1]
    auto concat_list = std::vector<Symbol>{policy_board, policy_pass};
    auto policy_concat = Concat("policy_concat", concat_list, concat_list.size());
#if MXNET_VERSION >= 10600
    auto policy_output = softmax("policy_softmax", policy_concat, Symbol());
#else
    auto policy_output = softmax("policy_softmax", policy_concat);
#endif

    // board value
    auto bv_black_slice = slice_axis("board_value_black_slice", board_value, 1, 1, dmlc::optional<int>(2));
    auto bv_black = Flatten("board_value_black_flatten", bv_black_slice);
    auto bv_output = bv_black * 2.0f - 1.0f;

    // state
    auto state_output = Flatten("state_flatten", state);

    return Symbol::Group({policy_output, bv_output, state_output});
}

// Symbol
Symbol ResNetSymbol(Symbol plane, Symbol policy_mcts, Symbol board_value_mcts, Symbol next_state, Symbol rate,
                    uint32_t num_filter, uint32_t tower_size, uint32_t img_size,
                    bool bottle_neck, std::string act_type, std::string norm_type, std::string se_type)
{
    auto input = Conv3x3_Norm_AC("input_conv", plane, num_filter, act_type, norm_type);
    auto tower = ResNetTower("res", input, num_filter, tower_size, bottle_neck, act_type, norm_type, se_type);
    return Head(tower, policy_mcts, board_value_mcts, next_state, rate, img_size, false);
}

Symbol ResNetV2Symbol(Symbol plane, Symbol policy_mcts, Symbol board_value_mcts, Symbol next_state, Symbol rate,
                      uint32_t num_filter, uint32_t tower_size, uint32_t img_size, bool bottle_neck,
                      std::string act_type, std::string norm_type, std::string se_type)
{
    auto input = Conv3x3("input_conv", plane, num_filter);
    auto tower = ResNetV2Tower("res", input, num_filter, tower_size, bottle_neck, act_type, norm_type, se_type);
    return Head(tower, policy_mcts, board_value_mcts, next_state, rate, img_size, true, act_type);
}

Symbol ResNeXtSymbol(Symbol plane, Symbol policy_mcts, Symbol board_value_mcts, Symbol next_state, Symbol rate,
                     uint32_t num_filter, uint32_t tower_size, uint32_t img_size, bool bottle_neck,
                     std::string act_type, std::string norm_type, std::string se_type)
{
    auto input = Conv3x3("input_conv", plane, num_filter);
    auto tower = ResNeXtTower("res", input, num_filter, tower_size, bottle_neck, act_type, norm_type, se_type);
    return Head(tower, policy_mcts, board_value_mcts, next_state, rate, img_size, true, act_type);
}

Symbol ResNetCZSymbol(Symbol plane, Symbol policy_mcts, Symbol board_value_mcts, Symbol next_state, Symbol rate,
                      uint32_t num_filter, uint32_t tower_size, uint32_t img_size, bool bottle_neck,
                      std::string act_type, std::string norm_type, std::string se_type)
{
    auto input = Conv3x3("input_conv", plane, num_filter);
    auto tower = ResNetCZTower("res", input, num_filter, tower_size, bottle_neck, act_type, norm_type, se_type);
    return Head(tower, policy_mcts, board_value_mcts, next_state, rate, img_size, true, act_type);
}

Symbol DPNSymbol(Symbol plane, Symbol policy_mcts, Symbol board_value_mcts, Symbol next_state, Symbol rate,
                 uint32_t num_filter, uint32_t tower_size, uint32_t img_size, uint32_t inc,
                 std::string act_type, std::string norm_type, std::string se_type)
{
    auto input = DPNInput("input", plane, num_filter, inc);
    auto tower = DPNTower("dpn", input, num_filter, tower_size, inc, act_type, norm_type, se_type);
    return Head(tower, policy_mcts, board_value_mcts, next_state, rate, img_size, true, act_type);
}

Symbol MixNetSymbol(Symbol plane, Symbol policy_mcts, Symbol board_value_mcts, Symbol next_state, Symbol rate,
                    uint32_t inc, uint32_t tower_size, uint32_t img_size, bool sparse,
                    std::string act_type, std::string norm_type, std::string se_type)
{
    auto input = MixNetInput("input", plane, inc);
    auto tower = MixNetTower("mix", input, inc, tower_size, sparse, act_type, norm_type, se_type);
    return Head(tower, policy_mcts, board_value_mcts, next_state, rate, img_size, true, act_type);
}

Symbol IterMixNetSymbol(Symbol plane, Symbol policy_mcts, Symbol board_value_mcts, Symbol next_state, Symbol rate,
                        uint32_t inc, uint32_t tower_size, uint32_t img_size, uint32_t iter,
                        std::string act_type, std::string norm_type, std::string se_type)
{
    auto input = IterMixNetInput("input", plane, inc, iter);
    auto tower = MixNetTower("iter_mix", input, inc, tower_size, false, act_type, norm_type, se_type);
    return Head(tower, policy_mcts, board_value_mcts, next_state, rate, img_size, true, act_type);
}

// Output
Symbol ResNetOutput(Symbol plane, uint32_t num_filter, uint32_t tower_size, bool bottle_neck,
                    std::string act_type, std::string norm_type, std::string se_type)
{
    auto input = Conv3x3_Norm_AC("input_conv", plane, num_filter, act_type, norm_type);
    auto tower = ResNetTower("res", input, num_filter, tower_size, bottle_neck, act_type, norm_type, se_type);
    return OutputHead(tower, false);
}

Symbol ResNetV2Output(Symbol plane, uint32_t num_filter, uint32_t tower_size, bool bottle_neck,
                      std::string act_type, std::string norm_type, std::string se_type)
{
    auto input = Conv3x3("input_conv", plane, num_filter);
    auto tower = ResNetV2Tower("res", input, num_filter, tower_size, bottle_neck, act_type, norm_type, se_type);
    return OutputHead(tower, true, act_type);
}

Symbol ResNetCZOutput(Symbol plane, uint32_t num_filter, uint32_t tower_size, bool bottle_neck,
                      std::string act_type, std::string norm_type, std::string se_type)
{
    auto input = Conv3x3("input_conv", plane, num_filter);
    auto tower = ResNetCZTower("res", input, num_filter, tower_size, bottle_neck, act_type, norm_type, se_type);
    return OutputHead(tower, true, act_type);
}

Symbol ResNeXtOutput(Symbol plane, uint32_t num_filter, uint32_t tower_size, bool bottle_neck,
                     std::string act_type, std::string norm_type, std::string se_type)
{
    auto input = Conv3x3("input_conv", plane, num_filter);
    auto tower = ResNeXtTower("res", input, num_filter, tower_size, bottle_neck, act_type, norm_type, se_type);
    return OutputHead(tower, true, act_type);
}

Symbol DPNOutput(Symbol plane, uint32_t num_filter, uint32_t tower_size, uint32_t inc,
                 std::string act_type, std::string norm_type, std::string se_type)
{
    auto input = DPNInput("input", plane, num_filter, inc);
    auto tower = DPNTower("dpn", input, num_filter, tower_size, inc, act_type, norm_type, se_type);
    return OutputHead(tower, true, act_type);
}

Symbol MixNetOutput(Symbol plane, uint32_t inc, uint32_t tower_size, bool sparse,
                    std::string act_type, std::string norm_type, std::string se_type)
{
    auto input = MixNetInput("input", plane, inc);
    auto tower = MixNetTower("mix", input, inc, tower_size, sparse, act_type, norm_type, se_type);
    return OutputHead(tower, true, act_type);
}

Symbol IterMixNetOutput(Symbol plane, uint32_t inc, uint32_t tower_size, uint32_t iter,
                        std::string act_type, std::string norm_type, std::string se_type)
{
    auto input = IterMixNetInput("input", plane, inc, iter);
    auto tower = MixNetTower("iter_mix", input, inc, tower_size, false, act_type, norm_type, se_type);
    return OutputHead(tower, true, act_type);
}
