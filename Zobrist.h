//
// Created by yuanyu on 2017.12.28.
//

#pragma once
#include <vector>

class Zobrist
{
public:
    static Zobrist& Get();
    void Init(size_t width, size_t height);

    uint64_t empty_board(){ return hash_empty_board; }// 返回空棋盘hash
    uint64_t black(int p){ return hash_black[p]; }
    uint64_t white(int p){ return hash_white[p]; }
    uint64_t ko(int p){ return hash_ko[p]; }
    uint64_t color(int c){ return hash_color[c]; }
    uint64_t move(int n){ return hash_move[n]; }


private:
    Zobrist();

private:
    std::vector<uint64_t> hash_black, hash_white, hash_ko, hash_move;
    uint64_t hash_empty_board;
    uint64_t hash_color[2];
};

#define GoHash (Zobrist::Get())
