//
// Created by yuanyu on 2018.05.26.
//

#ifndef SHIRO_SYMBOL_H
#define SHIRO_SYMBOL_H
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

// Basic
Symbol Conv(const std::string& name, Symbol data, Shape kernel, uint32_t num_filter, uint32_t num_group = 1,
            Shape stride = Shape(1, 1), Shape dilate = Shape(1, 1), Shape pad = Shape(0, 0));

Symbol IterConv(const std::string& name, Symbol data, Shape kernel, uint32_t iter, uint32_t num_filter, uint32_t num_group = 1,
                Shape stride = Shape(1, 1), Shape dilate = Shape(1, 1), Shape pad = Shape(0, 0));

Symbol Conv3x3(const std::string& name, Symbol data, uint32_t num_filter, uint32_t num_group = 1,
               Shape stride = Shape(1, 1), Shape dilate = Shape(1, 1), Shape pad = Shape(1, 1));
Symbol Conv1x1(const std::string& name, Symbol data, uint32_t num_filter, uint32_t num_group = 1,
               Shape stride = Shape(1, 1), Shape dilate = Shape(1, 1), Shape pad = Shape(0, 0));
Symbol Conv1x3(const std::string& name, Symbol data, uint32_t num_filter, uint32_t num_group = 1,
               Shape stride = Shape(1, 1), Shape dilate = Shape(1, 1), Shape pad = Shape(0, 1));
Symbol Conv3x1(const std::string& name, Symbol data, uint32_t num_filter, uint32_t num_group = 1,
               Shape stride = Shape(1, 1), Shape dilate = Shape(1, 1), Shape pad = Shape(1, 0));
Symbol Conv1x5(const std::string& name, Symbol data, uint32_t num_filter, uint32_t num_group = 1,
               Shape stride = Shape(1, 1), Shape dilate = Shape(1, 1), Shape pad = Shape(0, 2));
Symbol Conv5x1(const std::string& name, Symbol data, uint32_t num_filter, uint32_t num_group = 1,
               Shape stride = Shape(1, 1), Shape dilate = Shape(1, 1), Shape pad = Shape(2, 0));
Symbol Conv1x7(const std::string& name, Symbol data, uint32_t num_filter, uint32_t num_group = 1,
               Shape stride = Shape(1, 1), Shape dilate = Shape(1, 1), Shape pad = Shape(0, 3));
Symbol Conv7x1(const std::string& name, Symbol data, uint32_t num_filter, uint32_t num_group = 1,
               Shape stride = Shape(1, 1), Shape dilate = Shape(1, 1), Shape pad = Shape(3, 0));

Symbol IterConv3x3(const std::string& name, Symbol data, uint32_t iter, uint32_t num_filter, uint32_t num_group = 1,
                   Shape stride = Shape(1, 1), Shape dilate = Shape(1, 1), Shape pad = Shape(1, 1));

Symbol GN(const std::string& name, Symbol data, int num_groups = 1, float eps = 9.99999975e-06);
Symbol LN(const std::string& name, Symbol data, int axis = -1, float eps = 9.99999975e-06);
Symbol IN(const std::string& name, Symbol data, float eps = 00100000005);
Symbol BN(const std::string& name, Symbol data, double eps = 0.0010000000474974513, mx_float momentum = 0.899999976,
          bool fix_gamma = false, bool use_global_stats = false);
Symbol Norm(const std::string& name, Symbol data, std::string norm_type = "bn", std::string idx = "");

Symbol AC(const std::string& name, Symbol data, std::string act_type, std::string idx = "");
Symbol FC(const std::string& name, Symbol data, int num_hidden, bool no_bias = false, bool flatten = true);

// Block
Symbol Norm_AC(const std::string& name, Symbol data, std::string act_type, std::string norm_type, std::string idx = "");
Symbol Conv3x3_Norm_AC(const std::string& name, Symbol data, uint32_t num_filter, std::string act_type, std::string norm_type, std::string idx = "", uint32_t num_group = 1, Shape stride = Shape(1, 1));
Symbol Norm_AC_Conv3x3(const std::string& name, Symbol data, uint32_t num_filter, std::string act_type, std::string norm_type, std::string idx = "", uint32_t num_group = 1);
Symbol Norm_AC_Conv1x1(const std::string& name, Symbol data, uint32_t num_filter, std::string act_type, std::string norm_type, std::string idx = "", uint32_t num_group = 1);
Symbol SE_Block(const std::string& name, Symbol data, uint32_t num_filter);
Symbol SNSE_Block(const std::string& name, Symbol data, uint32_t num_filter, std::string norm_type);
Symbol SA_Block(const std::string& name, Symbol data);
Symbol CBAM_Block(const std::string& name, Symbol data, uint32_t num_filter);
Symbol SAA_Block(const std::string& name, Symbol data, uint32_t num_filter, std::string act_type, std::string norm_type);
Symbol KT_Block(const std::string& name, Symbol data, uint32_t num_filter, std::string act_type, std::string norm_type);
Symbol SE(const std::string& name, Symbol data, uint32_t num_filter,
          std::string act_type = "relu", std::string norm_type = "bn",
          std::string se_type = "", bool is_se_type_b = false);
Symbol ResBlock(const std::string& name, Symbol data, uint32_t num_filter,
                bool bottle_neck, std::string act_type, std::string norm_type, std::string se_type);
Symbol ResV2Block(const std::string& name, Symbol data, uint32_t num_filter, uint32_t blocks,
                  bool bottle_neck, std::string act_type, std::string norm_type, std::string se_type);
Symbol ResNeXtBlock(const std::string& name, Symbol data, uint32_t num_filter,
                    bool bottle_neck, std::string act_type, std::string norm_type, std::string se_type);

// Tower
Symbol ResNetTower(const std::string& name, Symbol data, uint32_t num_filter, uint32_t tower_size,
                   bool bottle_neck, std::string act_type, std::string norm_type, std::string se_type);
Symbol ResNetV2Tower(const std::string& name, Symbol data, uint32_t num_filter, uint32_t tower_size,
                     bool bottle_neck, std::string act_type, std::string norm_type, std::string se_type);
Symbol ResNeXtTower(const std::string& name, Symbol data, uint32_t num_filter, uint32_t tower_size,
                    bool bottle_neck, std::string act_type, std::string norm_type, std::string se_type);

Symbol ResNetCZTower(const std::string& name, Symbol data, uint32_t num_filter, uint32_t tower_size,
                     bool bottle_neck, std::string act_type, std::string norm_type, std::string se_type);

// DPN
std::vector<Symbol> DPNInput(const std::string& name, Symbol data, uint32_t num_filter, uint32_t inc);
std::vector<Symbol> DPNBlock(const std::string& name, std::vector<Symbol> input, uint32_t num_filter, uint32_t inc,
                             std::string act_type, std::string norm_type, std::string se_type);
Symbol DPNTower(const std::string& name, std::vector<Symbol> input, uint32_t num_filter, uint32_t tower_size, uint32_t inc,
                std::string act_type, std::string norm_type, std::string se_type);

// MixNet
std::vector<Symbol> MixNetInput(const std::string &name, Symbol data, uint32_t inc);
std::vector<Symbol> MixNetBlock(const std::string &name, Symbol data, uint32_t inc, bool sparse,
                                std::string act_type, std::string norm_type, std::string se_type);
Symbol MixNetTower(const std::string &name, std::vector<Symbol> input, uint32_t inc, uint32_t tower_size, bool sparse,
                   std::string act_type, std::string norm_type, std::string se_type);

// IterMixNet
std::vector<Symbol> IterMixNetInput(const std::string &name, Symbol data, uint32_t inc, uint32_t iter);
Symbol IterBlock(const std::string& name, Symbol data, uint32_t num_filter, uint32_t iter);

// Head
Symbol PolicyHead(Symbol data, bool norm_ac, std::string act_type, std::string norm_type);
Symbol PassHead(Symbol data, bool norm_ac, std::string act_type, std::string norm_type);
Symbol BoardValueHead(Symbol data, bool norm_ac, std::string act_type, std::string norm_type);
Symbol StateHead(Symbol data, bool norm_ac, std::string act_type, std::string norm_type);
Symbol Loss(Symbol policy, Symbol pass, Symbol board_value, Symbol state,
            Symbol policy_mcts, Symbol board_value_mcts, Symbol next_state, Symbol rate, uint32_t img_size);
Symbol Head(Symbol tower, Symbol policy_mcts, Symbol board_value_mcts, Symbol next_state, Symbol rate,
            uint32_t img_size, bool norm_ac, std::string act_type = "relu", std::string norm_type = "bn");
Symbol OutputHead(Symbol tower, bool norm_ac, std::string act_type = "relu", std::string norm_type = "bn");

// Symbol
Symbol ResNetSymbol(Symbol plane, Symbol policy_mcts, Symbol board_value_mcts, Symbol next_state, Symbol rate,
                    uint32_t num_filter, uint32_t tower_size, uint32_t img_size, bool bottle_neck,
                    std::string act_type = "relu", std::string norm_type = "bn", std::string se_type = "");

Symbol ResNetV2Symbol(Symbol plane, Symbol policy_mcts, Symbol board_value_mcts, Symbol next_state, Symbol rate,
                      uint32_t num_filter, uint32_t tower_size, uint32_t img_size, bool bottle_neck,
                      std::string act_type = "relu", std::string norm_type = "bn", std::string se_type = "");

Symbol ResNeXtSymbol(Symbol plane, Symbol policy_mcts, Symbol board_value_mcts, Symbol next_state, Symbol rate,
                     uint32_t num_filter, uint32_t tower_size, uint32_t img_size, bool bottle_neck,
                     std::string act_type = "relu", std::string norm_type = "bn", std::string se_type = "");

Symbol ResNetCZSymbol(Symbol plane, Symbol policy_mcts, Symbol board_value_mcts, Symbol next_state, Symbol rate,
                      uint32_t num_filter, uint32_t tower_size, uint32_t img_size, bool bottle_neck,
                      std::string act_type = "relu", std::string norm_type = "bn", std::string se_type = "");

Symbol DPNSymbol(Symbol plane, Symbol policy_mcts, Symbol board_value_mcts, Symbol next_state, Symbol rate,
                 uint32_t num_filter, uint32_t tower_size, uint32_t img_size, uint32_t inc,
                 std::string act_type = "relu", std::string norm_type = "bn", std::string se_type = "");

Symbol MixNetSymbol(Symbol plane, Symbol policy_mcts, Symbol board_value_mcts, Symbol next_state, Symbol rate,
                    uint32_t inc, uint32_t tower_size, uint32_t img_size, bool sparse = true,
                    std::string act_type = "relu", std::string norm_type = "bn", std::string se_type = "");

Symbol IterMixNetSymbol(Symbol plane, Symbol policy_mcts, Symbol board_value_mcts, Symbol next_state, Symbol rate,
                        uint32_t inc, uint32_t tower_size, uint32_t img_size, uint32_t iter, bool sparse = true,
                        std::string act_type = "relu", std::string norm_type = "bn", std::string se_type = "");

// Output
Symbol ResNetOutput(Symbol plane, uint32_t num_filter, uint32_t tower_size, bool bottle_neck,
                    std::string act_type = "relu", std::string norm_type = "bn", std::string se_type = "");

Symbol ResNetV2Output(Symbol plane, uint32_t num_filter, uint32_t tower_size, bool bottle_neck,
                      std::string act_type = "relu", std::string norm_type = "bn", std::string se_type = "");

Symbol ResNeXtOutput(Symbol plane, uint32_t num_filter, uint32_t tower_size, bool bottle_neck,
                     std::string act_type = "relu", std::string norm_type = "bn", std::string se_type = "");

Symbol ResNetCZOutput(Symbol plane, uint32_t num_filter, uint32_t tower_size, bool bottle_neck,
                      std::string act_type = "relu", std::string norm_type = "bn", std::string se_type = "");

Symbol DPNOutput(Symbol plane, uint32_t num_filter, uint32_t tower_size, uint32_t inc,
                 std::string act_type = "relu", std::string norm_type = "bn", std::string se_type = "");

Symbol MixNetOutput(Symbol plane, uint32_t inc, uint32_t tower_size, bool sparse = true,
                    std::string act_type = "relu", std::string norm_type = "bn", std::string se_type = "");

Symbol IterMixNetOutput(Symbol plane, uint32_t inc, uint32_t tower_size, uint32_t iter, bool sparse = true,
                        std::string act_type = "relu", std::string norm_type = "bn", std::string se_type = "");


#endif //SHIRO_SYMBOL_H
