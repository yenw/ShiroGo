//
// Created by yuanyu on 2017.12.28.
//

#pragma once
#include <vector>
#include <unordered_set>
#include <array>
#include "config.h"
using std::vector;
using std::unordered_set;

class GoBoard
{
public:
    GoBoard(int line);
    GoBoard(int line_x, int line_y);
    void init(int line_x, int line_y);
    void board_show();
    void mv_show(vector<int64_t>& mv, uint64_t N);

    // 平面相关
    void update_planes();
    void get_planes(vector<float>& stone, int history);
    void get_planes01(vector<float>& stone, int history, int mode, int shuffle, int p, int selu);
    void get_state(vector<float>& state);

    // 交换平面中的动作
    bool swap_action();
    static vector<int> calc_action(const vector<float>& stone, int history, int size);
    static vector<int> shuffle_action(vector<int>& action);

    // 路数
    int get_width(){ return xline; }
    int get_height(){ return yline; }

    // 颜色相关
    void set_turn_color(int color) { turn_color = color; }
    int get_turn_color() { return turn_color; }
    bool next_black() { return turn_color == BLACK; }

    // 大棋盘坐标与小棋盘(x, y)转换
    int pos(int x, int y) { return (x + 1) + (y + 1) * (xline + 2); }
    int posx(int p) { return p % (xline + 2) - 1; }
    int posy(int p) { return p / (xline + 2) - 1; }
    int pos_no_pad(int p){ return posx(p) + posy(p) * xline; }

    // 死子相关
    int get_dead_count() { return dead_count; }
    int get_dead_black() { return dead_black; }
    int get_dead_white() { return dead_white; }

    // 获取棋盘信息
    int move_count(){ return m_move_count; }
    int move_count_max(){ return xline * yline * 3; }
    int pass_count(){ return pass; }
    int color(int p) { return board[p].color; }
    int other_color(int c) { return c ^ 1; }
    int id_head(int p) { return board[p].id_head; }
    int id_next(int p) { return board[p].id_next; }
    int liberty(int p) { return board[id_head(p)].liberty; }
    int stone_count(int p) { return board[id_head(p)].stone_count; }

    // 包装函数, 防止自己写时忘记 id_head(p)
    void liberty_add(int p) { ++board[id_head(p)].liberty; }
    void liberty_sub(int p) { --board[id_head(p)].liberty; }
    void liberty_plus(int p, int lib) { board[id_head(p)].liberty += lib; }
    void liberty_set(int p, int new_lib) { board[id_head(p)].liberty = new_lib; }
    void stone_plus(int p1, int p2) { board[id_head(p1)].stone_count += board[id_head(p2)].stone_count; }

    // 简要判断函数
    bool on_board(int p) { return color(p) != GRAY; }
    bool on_board_color(int c) { return c != GRAY; }// 加速
    bool is_black(int c) { return c == BLACK; }
    bool is_white(int c) { return c == WHITE; }
    bool is_empty(int p) { return color(p) == EMPTY; }
    bool is_empty_color(int c) { return c == EMPTY; }// 加速

    bool is_self(int p, int c) { return color(p) == c; } // p处棋串颜色是否为c
    bool is_not_self(int p, int c) { return color(p) != c; }
    bool is_self_color(int c1, int c2) { return c1 == c2; } // 加速
    bool is_not_self_color(int c1, int c2) { return c1 != c2; }

    bool is_foe(int p, int c) { return color(p) == other_color(c); }
    bool is_not_foe(int p, int c) { return color(p) != other_color(c); }
    bool is_foe_color(int c1, int c2) { return c1 == other_color(c2); }// 加速
    bool is_not_foe_color(int c1, int c2) { return c1 != other_color(c2); }

    bool is_stone(int p) { int c = color(p); return c == BLACK || c == WHITE; }
    bool is_not_stone(int p) { int c = color(p); return c != BLACK && c != WHITE; }
    bool is_stone_color(int c) { return c == BLACK || c == WHITE; }
    bool is_not_stone_color(int c) { return c != BLACK && c != WHITE; }

    bool is_pass(int p) { return p == PASS_MOVE; }
    bool is_ko_point(int p) { return p == ko_pos; }// p处是打劫禁手吗?
    //bool is_atari(int p) { return liberty(p) == 1; } // p处棋串能否被提吃, 不考虑打劫和禁同
    bool is_adjacent(int g, int p); // p是否与g相邻
    bool no_move(int p) { return p == NO_MOVE; }
    //bool legal_pass() { return m_move_count * 2 >= xline * yline; }// 禁止过早pass
    bool legal_pass() { return true; }
    bool max_move(){ return m_move_count >= move_count_max(); }
    bool game_over() { return pass_count() >= 2 || max_move(); }

    // 标记,
    // 使用时一定要注意, 不能被多次new_mask 即父子函数均存在new_mask
    bool unmask(int p) { return board_mask[p] != next_mask; } // 测试点
    void mask(int p) { board_mask[p] = next_mask; } // 标记点
    bool unmask_group(int p) { return board_mask[id_head(p)] != next_mask; } // 测试棋串
    void mask_group(int p) { board_mask[id_head(p)] = next_mask; } // 标记棋串
    void new_mask() { ++next_mask; }

    // 试下后信息获取
    bool try_suicide(int p, int c);// 下在p处 是否是自杀
    bool try_suicide_foe(int p, int c); // c的对手下在p处 是否是自杀
    bool try_superko(int p, int c);

    // 气相关函数
    int common_liberty_count(int p1, int p2);// p1处棋串, 与p2处棋串的公气
    vector<int> common_liberty(int p1, int p2);// p1处棋串, 与p2处棋串的公气
    vector<int> find_liberty(int p);// 返回p处棋串的所有气

    // 下棋相关函数
    void recalc_liberty(int p);// 重新计算p处棋串的气, 并调用liberty_set(p, new_liberty);
    void group_capture(int p);// 提掉p处棋串
    void group_merge(int p1, int p2, int common_lib);// 连接两个棋组, 若只有两个, 可事先算好公气.
    void ko_point(int p);// 由当前落子点(p), 计算禁手点

    void move_at(int p){ move_at(p, get_turn_color()); }
    void move_at(int p, int c); // 落子, 实则由play调用
    bool is_legal(int p){ return is_legal(p, get_turn_color()); }
    bool is_legal(int p, int c);// 下在p处 是否有效
    bool play(int p){ return play(p, get_turn_color()); }
    bool play(int p, int c);// p处合法, 则落子

    // 分数相关, 黑视角
    void tt_reach(vector<int>& reach_board, int reach_color);
    vector<double> score_bv();
    float score_bonus(vector<double>& board_value, float bonus);
    int score_tt(){ vector<int64_t> tmp; return score_tt(tmp); }
    int score_tt(vector<int64_t>& mv);
    int score_stone();

    vector<int> empty_point(){ return vector<int>(ep.begin(), ep.begin() + ep_size); }

    // hash
    void hash_update(int p, int c);
    uint64_t hash_uct();
    uint64_t hash_board(){ return Zhash; }

    enum Color
    {
        WHITE,
        BLACK,
        EMPTY,
        GRAY
    };

    enum move
    {
        NO_MOVE = -3,
        PASS_MOVE = -2
    };

    struct Group
    {
        int color;
        int id_head;
        int id_next;
        int liberty;
        int stone_count;
        int ep;
    };

private:
    // 棋盘参数
    int xline, yline;
    int board_size, board_min, board_max;

    // 死子
    int dead_count, dead_black, dead_white;

    // 棋盘状态信息
    vector<Group> board;

    // 当前落子方
    int turn_color;

    int m_move_count;

    // 打劫禁招点
    int ko_pos;

    // 上一步落子, 显示用
    int last_p;

    // PASS
    int pass;

    // 标记专用
    vector<int> board_mask;
    int next_mask;

    // 空点
    vector<int> ep;// empty pos
    int ep_size;

    // Up, Left, Right, Down
    vector<int> delta;
    int delta_diag, delta_fly;

    // Zobrist Hash
    uint64_t Zhash;// 状态Hash, 不包含手数, 下一落子方, 禁招点等信息
    unordered_set<uint64_t> Zhash_history;

    // Network planes
    vector<vector<float>> black_planes;
    vector<vector<float>> white_planes;
    vector<vector<float>> total_planes;
    vector<float> zero_plane;
    vector<float> one_plane;

    // take
    std::array<bool, INPUT_HISTORY> take_history{};// 记录array[step % INPUT_HISTORY], !(提子&&弃权)
};
