//
// Created by yuanyu on 2018.01.30.
//

#pragma once
#include <cmath>

void transform_xy(int& x, int& y, int width, int height, int mode);
int transform_p(int p, int width, int height, int mode);

template <typename T1, typename T2>
void transform_policy(T1* dst, T2* src, int width, int height, int mode = 1)
{
    int p, new_p;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            p = x + y * width;
            new_p = transform_p(p, width, height, mode);
            dst[new_p] = src[p];
        }
    }
    dst[width * height] = src[width * height];// pass move
}

template <typename T1, typename T2>
void transform_board_value(T1* dst, T2* src, int width, int height, int mode, bool shuffle)
{
    int p, new_p;
    int sign = shuffle ? -1: 1;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            p = x + y * width;
            new_p = transform_p(p, width, height, mode);
            dst[new_p] = src[p] * sign;
        }
    }
}

template <typename T1, typename T2>
void transform_stone(T1* dst, T2* src, int planes, int width, int height, int mode, bool shuffle)
{
    int board_size = width * height;
    int p, new_p;
    if ( !shuffle )
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                p = x + y * width;
                new_p = transform_p(p, width, height, mode);
                for (int i = 0; i < planes; ++i)
                    dst[new_p + i * board_size] = src[p + i * board_size];
            }
        }
    }
    else
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                p = x + y * width;
                new_p = transform_p(p, width, height, mode);
                for (int i = 0; i < planes; i += 2)
                    dst[new_p + i * board_size] = src[p + (i + 1) * board_size];

                for (int i = 0; i < planes; i += 2)
                    dst[new_p + (i + 1) * board_size] = src[p + i * board_size];
            }
        }
    }
}

template <typename T1>
void transform_stone_swap(T1* src, int* action_old, int* action_new, int action_size, int planes, int width, int height, int mode, bool shuffle)
{
    // 根据mode和shuffle, 转换action和shuffle_action
    for (int i = 0; i < action_size; ++i)
    {
        auto color = (action_old[i] & 1) ^ shuffle;
        action_old[i] = (transform_p(action_old[i] >> 1, width, height, mode) << 1) | color;
        action_new[i] = (transform_p(action_new[i] >> 1, width, height, mode) << 1) | color;
    }

    // 交换动作
    int board_size = width * height;
    int max_action_size = planes / 2;
    int begin_i = max_action_size - action_size;

    for (int i = begin_i; i < max_action_size; ++i)
    {
        // remove action_old[begin_i, i]
        for (int k = 0; k <= i - begin_i; ++k)
        {
            auto* board = &src[i * 2 * board_size];
            if (action_old[k] & 1)// black
                board[(action_old[k] >> 1) + board_size] = 0.0f;
            else
                board[action_old[k] >> 1] = 0.0f;
        }

        // add action_new[begin_i, i]
        for (int k = 0; k <= i - begin_i; ++k)
        {
            auto* board = &src[i * 2 * board_size];
            if (action_new[k] & 1)// black
                board[(action_new[k] >> 1) + board_size] = 1.0f;
            else
                board[action_new[k] >> 1] = 1.0f;
        }
    }
}

template <typename T1, typename T2>
void transform_action(T1* dst, T2* src, int planes, int width, int height, int mode, bool no_action)
{
    int board_size = width * height;
    if ( !no_action )
    {
        int p, new_p;
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                p = x + y * width;
                new_p = transform_p(p, width, height, mode);
                for (int i = 0; i < planes; ++i)
                    dst[new_p + i * board_size] = src[p + i * board_size];
            }
        }
    }
    else
    {
        if (planes == 2)
        {
            for (int p = 0; p < width * height; ++p)
            {
                dst[p] = 0.0f;
                dst[p + board_size] = 1.0f;
            }
        }
        else if (planes == 1)
        {
            for (int p = 0; p < width * height; ++p)
                dst[p] = 0.0f;
        }
    }
}

template <typename T1, typename T2>
void transform_state(T1* dst, T2* src_act, T2* src_no_act, int width, int height, int mode, bool shuffle, bool no_action)
{
    int board_size = width * height;
    int p, new_p;
    if ( !shuffle )
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                p = x + y * width;
                new_p = transform_p(p, width, height, mode);
                if ( !no_action )
                {
                    for (int i = 0; i < 3; ++i)
                        dst[new_p + i * board_size] = src_act[p + i * board_size];
                }
                else
                {
                    dst[new_p + 0 * board_size] = src_no_act[p + 0 * board_size];// WHITE
                    dst[new_p + 1 * board_size] = src_no_act[p + 1 * board_size];// BLACK
                    if (src_no_act[p + 0 * board_size] == 0.0f && src_no_act[p + 1 * board_size] == 0.0f)
                        dst[new_p + 2 * board_size] = 1.0f;// EMPTY
                    else
                        dst[new_p + 2 * board_size] = 0.0f;// EMPTY
                }
            }
        }
    }
    else
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                p = x + y * width;
                new_p = transform_p(p, width, height, mode);
                if ( !no_action )
                {
                    dst[new_p + 0 * board_size] = src_act[p + 1 * board_size];// WHITE -> BLACK
                    dst[new_p + 1 * board_size] = src_act[p + 0 * board_size];// BLACK -> WHITE
                    dst[new_p + 2 * board_size] = src_act[p + 2 * board_size];// EMPTY
                }
                else
                {
                    dst[new_p + 0 * board_size] = src_no_act[p + 1 * board_size];// WHITE -> BLACK
                    dst[new_p + 1 * board_size] = src_no_act[p + 0 * board_size];// BLACK -> WHITE
                    if (src_no_act[p + 0 * board_size] == 0.0f && src_no_act[p + 1 * board_size] == 0.0f)
                        dst[new_p + 2 * board_size] = 1.0f;// EMPTY
                    else
                        dst[new_p + 2 * board_size] = 0.0f;// EMPTY
                }
            }
        }
    }
}

template <typename T1>
void transform_selu(T1* src, int planes, int width, int height)
{
    int board_size = height * width;
    for (int pl = 0; pl < planes; ++pl)
    {
        T1* board = &src[pl * board_size];
        // mu
        float mu = 0.0f;
        for (int i = 0; i < board_size; ++i)
            mu += board[i];

        if (mu == 0.0f)
            continue;

        mu /= board_size;

        // std_dev
        float v = 0.0f;
        float var = 0.0f;
        for (int i = 0; i < board_size; ++i)
        {
            v = board[i] - mu;
            var += v * v;
        }
        var /= board_size;
        float std_dev = 1.0f / std::sqrt(var + 1e-8f);

        // normal
        for (int i = 0; i < board_size; ++i)
            board[i] = std_dev * (board[i] - mu);
    }
}