//
// Created by yuanyu on 2020.02.01.
//

#ifndef SHIRO_CONFIG_H
#define SHIRO_CONFIG_H

#include <string>
#include <cmath>

const int INPUT_HISTORY = 8;
const int COLOR_PLANES = 2;
const int ACTION_PLANES = 2;// PP: 0

const std::string FOLDER_DATA = "data";
const std::string FOLDER_RD = "rd";
const std::string FOLDER_SGF = "sgf";
const std::string FOLDER_MODEL = "model";
const std::string ACT_TYPE{"gelu"};// PP: relu

const bool USE_RD = true;// PP: false
const bool USE_SWAP_ACTION = true;// PP: false
const float ITER_RATE = 0.25 * pow(1.2f, ACTION_PLANES * 0.5 + USE_RD + USE_SWAP_ACTION);
const int SP_THREADS = 4;// 模拟n线程各对弈1局, 实际上是单线程.
#define MXNET_VERSION 10600
//#define MXNET_PR

#endif //SHIRO_CONFIG_H
