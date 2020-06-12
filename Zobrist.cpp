//
// Created by yuanyu on 2017.12.28.
//

#include <unordered_set>
#include <random>
#include "Random.h"
#include "Zobrist.h"

using std::unordered_set;

Zobrist& Zobrist::Get()
{
    static Zobrist z;
    return z;
}

Zobrist::Zobrist()
{
    Init(19, 19);
}

void Zobrist::Init(size_t width, size_t height)
{
    auto board_size = (width + 2) * (height + 2);
    hash_black.resize(board_size, 0);
    hash_white.resize(board_size, 0);
    hash_ko.resize(board_size, 0);
    hash_move.resize(width * height * 3, 0);

    unordered_set<uint64_t> uset;
    std::mt19937_64 mt(time(0));
    for (int i = 0; i < board_size * 3 + 3 + width * height * 3;)// 生成不重复64bit随机数
    {
        auto r = mt();
        if ( uset.find(r) == uset.end() )
        {
            uset.insert(r);
            ++i;
        }
    }

    auto hash_it = uset.begin();
    for (int i = 0; i < board_size; ++i)// 填充
    {
        hash_black[i] = *hash_it++;
        hash_white[i] = *hash_it++;
        hash_ko[i] = *hash_it++;
    }

    hash_empty_board = *hash_it++;
    hash_color[0] = *hash_it++;
    hash_color[1] = *hash_it++;

    for (int i = 0; i < width * height * 3; ++i)// 填充
        hash_move[i] = *hash_it++;
}