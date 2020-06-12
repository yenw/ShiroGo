//
// Created by yuanyu on 2017.12.28.
//

#include <iostream>
#include <random>
#include <algorithm>
#include "GoBoard.h"
#include "Transform.h"
#include "Random.h"
#include "Zobrist.h"
#include "config.h"

using std::cout;
using std::cin;
using std::cerr;
using std::endl;

inline bool find(const int* array, int size, int value)
{
    for (int i = 0; i < size; ++i)
    {
        if (array[i] == value)
            return true;
    }

    return false;
}

inline bool insert(int* array, int& size, int value)
{
    for (int i = 0; i < size; ++i)
    {
        if (array[i] == value)
            return false;
    }

    array[size++] = value;
    return true;
}

GoBoard::GoBoard(int line)
{
    init(line, line);
}

GoBoard::GoBoard(int line_x, int line_y)
{
    init(line_x, line_y);
}

void GoBoard::init(int line_x, int line_y)
{
    int offboard_line = 1;
    xline = line_x;
    yline = line_y;
    board_size = (xline + 2) * (yline + 2);
    board_min = pos(0, 0);
    board_max = board_size - board_min;
    board.resize(board_size);
    board_mask.resize(board_size, 0);
    ep.resize(xline * yline, 0);

    pass = 0;
    m_move_count = 0;
    turn_color = BLACK;
    ko_pos = NO_MOVE;
    last_p = NO_MOVE;
    next_mask = 0;
    dead_count = dead_black = dead_white = 0;

    // 初始化棋盘
    for (int i = 0; i < board_size; ++i)
    {
        board[i].color = GRAY;
        board[i].id_head = i;
        board[i].id_next = i;
        board[i].liberty = 0;
        board[i].stone_count = 0;
    }

    ep_size = 0;
    for (int y = 0; y < yline; ++y)
    {
        for (int x = 0; x < xline; ++x)
        {
            int p = pos(x, y);
            board[p].color = EMPTY;
            board[p].id_head = p;
            board[p].id_next = p;
            board[p].liberty = 0;
            board[p].stone_count = 1;
            board[p].ep = ep_size;

            ep[ep_size++] = p;// 空点记录
        }
    }

    /* 坐标如下图所示
      4  0  5
      3  p  1
      7  2  6
    */

    //delta
    delta.clear();

    // Up, Right, Down, Left
    delta.push_back(-(xline + 2));
    delta.push_back(1);
    delta.push_back(xline + 2);
    delta.push_back(-1);

    delta_diag = 4;
    delta.push_back(delta[0] + delta[3]);// Up-Left
    delta.push_back(delta[0] + delta[1]);// Up-Right
    delta.push_back(delta[2] + delta[1]);// Down-Right
    delta.push_back(delta[2] + delta[3]);// Down-Left

    delta_fly = 8;

    // 空棋盘初始hash
    Zhash = GoHash.empty_board();

    // Network
    black_planes = vector<vector<float>>();
    white_planes = vector<vector<float>>();
    total_planes = vector<vector<float>>();

    zero_plane = vector<float>(xline * yline, 0.0f);
    one_plane = vector<float>(xline * yline, 1.0f);
}

bool GoBoard::is_legal(int p, int c)
{
    if ( is_pass(p) )
        return true;

    int cp = color(p);

    // not onBoard
    if ( !on_board_color(cp) )
        return false;

    // not empty
    if ( !is_empty_color(cp) )
        return false;

    // Ko
    if (   is_ko_point(p)
        && turn_color == c)// 若白方提劫, 黑弃权, 没这行的话白不能粘劫
    {
        return false;
    }

    // suicude
    if ( try_suicide(p, c) )
        return false;

    // superko
    if ( try_superko(p, c) )
        return false;

    return true;
}

bool GoBoard::is_adjacent(int g, int p)
{
    for (int i = 0; i < 4; ++i)
    {
        if (g == id_head(p + delta[i]))
            return true;
    }

    return false;
}

bool GoBoard::try_suicide(int p, int c)
{
    // empty or (is_self && !is_atari) or (is_foe && is_atari)
    // return false

    // Up, Left, Right, Down
    for (int c_delta, p_delta, i = 0; i < 4; ++i)
    {
        p_delta = p + delta[i];
        c_delta = color(p_delta);
        if ( is_empty_color(c_delta) )
            return false;

        if (    on_board_color(c_delta)
            && (is_self_color(c_delta, c) ^ (liberty(p_delta) == 1)))
        {
            return false;
        }
    }

    return true;
}

bool GoBoard::try_suicide_foe(int p, int c)
{
    return try_suicide(p, other_color(c));
}

bool GoBoard::try_superko(int p, int c)
{
    auto save_hash = Zhash;
    hash_update(p, c);// 更新落子

    int id_foe[4];
    int id_foe_size = 0;
    for (int c_delta, p_delta, i = 0; i < 4; ++i)
    {
        p_delta = p + delta[i];
        c_delta = color(p_delta);
        if ( !on_board_color(c_delta) )
            continue;

        if ( is_foe_color(c_delta, c) )
            insert(id_foe, id_foe_size, id_head(p_delta));// 去重的插入
    }

    // 处理相邻对方棋组
    for (int foe, i = 0; i < id_foe_size; ++i)
    {
        foe = id_foe[i];
        if (liberty(foe) == 1)
        {
            // 模拟提掉对方棋组
            int c_foe = color(foe);
            int head = id_head(foe);
            int next = head;
            int record_next;
            do
            {
                // update hash
                hash_update(next, c_foe);

                //next
                next = id_next(next);
            }
            while (next != head);
        }
    }

    // 恢复hash
    std::swap(Zhash, save_hash);

    // 查重
    if ( Zhash_history.find(save_hash) != Zhash_history.end() )
        return true;

    return false;
}

int GoBoard::common_liberty_count(int p1, int p2)
{
    int common = 0;
    if ( liberty(p1) < liberty(p2) )
        std::swap(p1, p2);

    // mask
    new_mask();

    int head = id_head(p1);
    int head2 = id_head(p2);
    int next = head;
    do
    {
        //add liberty(neighbor) Up, Left, Right, Down
        for (int p_delta, i = 0; i < 4; ++i)
        {
            p_delta = next + delta[i];
            if (   is_empty(p_delta)
                   && unmask(p_delta))
            {
                if ( is_adjacent(head2, p_delta) )
                    ++common;

                mask(p_delta);
            }
        }

        //next
        next = id_next(next);
    }
    while(next != head);

    return common;
}

void GoBoard::recalc_liberty(int p)
{
    new_mask();
    int new_liberty = 0;

    int head = id_head(p);
    int next = head;
    do
    {
        //add liberty(neighbor) Up, Left, Right, Down
        for (int p_delta, i = 0; i < 4; ++i)
        {
            p_delta = next + delta[i];
            if (   is_empty(p_delta)
                && unmask(p_delta) )
            {
                ++new_liberty;
                mask(p_delta);
            }
        }

        //next
        next = id_next(next);
    }
    while(next != head);

    liberty_set(p, new_liberty);
}


void GoBoard::ko_point(int p)
{
    if (   dead_count == 1
        && stone_count(p) == 1
        && liberty(p) == 1)
    {
        for (int p_delta, i = 0; i < 4; ++i)
        {
            p_delta = p + delta[i];
            if ( is_empty(p_delta) )
            {
                ko_pos = p_delta;
                return;
            }
        }
    }

    ko_pos = NO_MOVE;
}

void GoBoard::group_capture(int p)
{
    dead_count += stone_count(p);
    int c = color(p);
    int head = id_head(p);
    int next = head;
    int record_next;
    do
    {
        //add liberty(neighbor) Up, Left, Right, Down
        new_mask();// 用来记录增加气的棋组, 防止多次增加
        for (int p_delta, i = 0; i < 4; ++i)
        {
            p_delta = next + delta[i];
            if (   is_foe(p_delta, c)
                && unmask_group(p_delta))
            {
                mask_group(p_delta);
                liberty_add(p_delta);
            }
        }

        // update hash
        hash_update(next, c);

        // record
        record_next = id_next(next);// 必要, 不可省

        // update board
        board[next].color = EMPTY;
        board[next].id_head = next;
        board[next].id_next = next;
        board[next].liberty = 0;
        board[next].stone_count = 1;

        // 更新空点
        board[next].ep = ep_size;
        ep[ep_size] = next;
        ++ep_size;

        //next
        next = record_next;
    }
    while(next != head);
}

void GoBoard::group_merge(int p1, int p2, int common_lib)
{
    if (id_head(p1) == id_head(p2))
        return;

    int head, head_little, next, last;
    int pos_many, pos_little;
    if (stone_count(p1) > stone_count(p2))
    {
        pos_many = p1;
        pos_little = p2;
    }
    else
    {
        pos_many = p2;
        pos_little = p1;
    }

    // 更新气
    liberty_plus(pos_many, liberty(pos_little) - 1 - common_lib);

    // 更新棋子数量
    stone_plus(pos_many, pos_little);

    head = id_head(pos_many);
    head_little = id_head(pos_little);
    next = head_little;
    do
    {
        board[next].id_head = head;
        last = next;
        next = id_next(next);
    }
    while(next != head_little);
    board[last].id_next = id_next(head);
    board[head].id_next = head_little;
}

void GoBoard::move_at(int p, int c)
{
    dead_count = 0;
    ++m_move_count;
    if ( is_pass(p) )
    {
        ++pass;
        take_history[(m_move_count - 1) % INPUT_HISTORY] = true;

        ko_pos = NO_MOVE;
        turn_color = other_color(c);
        last_p = p;
        update_planes();
        return;
    }
    else
        pass = 0;

    // 更新hash
    hash_update(p, c);// 更新当前落子hash

    // 更新空点
    int ep_play = board[p].ep;
    int ep_max = ep[--ep_size];
    board[ep_max].ep = ep_play;// 把下子的位置抢过来 board --> ep
    ep[ep_play] = ep_max;// ep --> board

    // 更新自己
    // id_head, id_next, stone_count, liberty 已初始化
    board[p].color = c;

    // 获取上下左右信息
    int id_self[4];
    int id_self_size = 0;

    int id_foe[4];
    int id_foe_size = 0;
    for (int c_delta, p_delta, i = 0; i < 4; ++i)
    {
        p_delta = p + delta[i];
        c_delta = color(p_delta);
        if ( !on_board_color(c_delta) )
            continue;

        if ( is_foe_color(c_delta, c) )
            insert(id_foe, id_foe_size, id_head(p_delta));// 去重的插入
        else
        if ( is_self_color(c_delta, c) )
            insert(id_self, id_self_size, id_head(p_delta));// 去重的插入
        else
            liberty_add(p);
    }

    // 处理相邻对方棋组
    for (int foe, i = 0; i < id_foe_size; ++i)
    {
        foe = id_foe[i];
        liberty_sub(foe);
        if (liberty(foe) == 0)
            group_capture(foe);
    }

    // 处理相邻己方棋组
    switch (id_self_size)
    {
        case 1:
            group_merge(id_self[0], p, common_liberty_count(id_self[0], p));
            break;
        case 2:// 特定情况可特殊化处理
        case 3:
        case 4:
            for (int i = 0; i < id_self_size; ++i)
                group_merge(id_self[i], p, 0);

            recalc_liberty(p);
        default:
            break;
    }

    // ko
    ko_point(p);

    // dead
    if (c == BLACK)
        dead_white += dead_count;
    else
        dead_black += dead_count;

    turn_color = other_color(c);
    last_p = p;
    Zhash_history.insert(Zhash);

    update_planes();
    take_history[(m_move_count - 1) % INPUT_HISTORY] = (dead_count > 0);
}

bool GoBoard::play(int p, int c)
{
    if ( !is_legal(p, c) )
        return false;

    move_at(p, c);
    return true;
}

void GoBoard::tt_reach(vector<int>& reach_board, int reach_color)
{
    vector<int> tt_color(reach_board.size(), -1);
    int ti = 0;
    for (size_t p = 0; p < board.size(); ++p)
    {
        if (color(p) == reach_color)
        {
            tt_color[ti++] = p;
            reach_board[p] = reach_color;
        }
    }

    int p;
    while ( ti != 0 )
    {
        --ti;
        p = tt_color[ti];
        for (int p_delta, i = 0; i < 4; ++i)
        {
            p_delta = p + delta[i];
            if (   is_empty_color(reach_board[p_delta])
                   && is_empty(p_delta))
            {
                reach_board[p_delta] = reach_color;
                tt_color[ti++] = p_delta;
            }
        }
    }
}

int GoBoard::score_tt(vector<int64_t>& mv)
{
    if (mv.size() != board.size())
    {
        mv.clear();
        mv.resize(board.size(), 0);
    }

    vector<int> tt_black(board.size(), EMPTY);
    vector<int> tt_white(board.size(), EMPTY);
    tt_reach(tt_black, BLACK);
    tt_reach(tt_white, WHITE);
    int tt = 0;
    for (size_t p = 0, c; p < board.size(); ++p)
    {
        c = color(p);
        if ( is_empty_color(c) )
        {
            if ( !is_empty_color(tt_black[p]) )
            {
                if ( is_empty_color(tt_white[p]) )
                {
                    ++tt;
                    ++mv[p];
                }

            }
            else if ( !is_empty_color(tt_white[p]) )
            {
                --tt;
                --mv[p];
            }
        }
        else if ( is_black(c) )
        {
            ++tt;
            ++mv[p];
        }
        else if ( is_white(c) )
        {
            --tt;
            --mv[p];
        }
    }

    //tt -= is_white(get_turn_color());
    return tt;
}

vector<double> GoBoard::score_bv()
{
    vector<double> board_value(get_width() * get_height(), 0.0);
    vector<int> tt_black(board.size(), EMPTY);
    vector<int> tt_white(board.size(), EMPTY);
    tt_reach(tt_black, BLACK);
    tt_reach(tt_white, WHITE);
    for (size_t p = 0, np, c; p < board.size(); ++p)
    {
        c = color(p);
        np = pos_no_pad(p);
        if ( is_empty_color(c) )
        {
            if ( !is_empty_color(tt_black[p]) )
            {
                if ( is_empty_color(tt_white[p]) )
                    board_value[np] = 1.0;//Black

            }
            else if ( !is_empty_color(tt_white[p]) )
            {
                board_value[np] = -1.0;//White
            }
        }
        else if ( is_black(c) )
        {
            board_value[np] = 1.0;//Black
        }
        else if ( is_white(c) )
        {
            board_value[np] = -1.0;//White
        }
    }

    return board_value;
}

float GoBoard::score_bonus(vector<double>& board_value, float bonus)
{
    if (bonus == 0.0f)
        return 0.0f;

    auto value = 0.0f;
    auto half_bonus = bonus * 0.5f;
    for (size_t p = 0, c, np; p < board.size(); ++p)
    {
        c = color(p);
        if ( is_black(c) )
        {
            np = pos_no_pad(p);
            value += (board_value[np] - 1.0) * half_bonus;// 黑子: -1(白占据) -> -bonus, 1(黑占据) -> -0
        }
        else if ( is_white(c) )
        {
            np = pos_no_pad(p);
            value += (board_value[np] + 1.0) * half_bonus;// 白子: -1(白占据) -> +0, 1(黑占据) -> +bonus
        }
    }
    return value + float(dead_white - dead_black) * bonus;
}

int GoBoard::score_stone()
{
    int stone = 0;
    for (size_t p = 0, c; p < board.size(); ++p)
    {
        c = color(p);
        if ( is_black(c) )
            ++stone;
        else if ( is_white(c) )
            --stone;
    }

    return stone;
}

void GoBoard::hash_update(int p, int c)
{
    if (c == BLACK)
        Zhash ^= GoHash.black(p);
    else
        Zhash ^= GoHash.white(p);
}

uint64_t GoBoard::hash_uct()
{
    // 提供UCT搜索的Hash
    // 包含棋盘信息: 棋子, 下一手, 对局长度, 打劫位置.
    // 任意一个不同, 视为不同局面.
    auto uct = Zhash;
    uct ^= GoHash.color(get_turn_color());
    uct ^= GoHash.move(move_count());
    if (ko_pos != NO_MOVE)
        uct ^= GoHash.ko(ko_pos);

    return uct;
}

void GoBoard::board_show()
{
    int x, y;
    int board_width = xline;
    int board_height = yline;

    int corner_x = (board_width >= 10) ? 3: 2;
    int corner_y = (board_height >= 10) ? 3: 2;

    vector<int> star_x;
    vector<int> star_y;
    if (board_width >= 7 && board_height >= 7)
    {
        star_x.push_back(corner_x);
        star_y.push_back(corner_y);

        star_x.push_back(board_width - corner_x - 1);
        star_y.push_back(corner_y);

        star_x.push_back(corner_x);
        star_y.push_back(board_height - corner_y - 1);

        star_x.push_back(board_width - corner_x - 1);
        star_y.push_back(board_height - corner_y - 1);

        int side_x = (board_width - 1) / 2 - corner_x;
        int side_y = (board_height - 1) / 2 - corner_y;
        if (board_width % 2 != 0 && board_height % 2 != 0 && side_x >= 3 && side_y >= 3)
        {
            star_x.push_back((board_width - 1) / 2);
            star_y.push_back((board_height - 1) / 2);
        }

        if (board_width % 2 != 0 && side_x >= 3 && side_y >= 3 && side_x + side_y >=7)
        {
            star_x.push_back((board_width - 1) / 2);
            star_y.push_back(corner_y);

            star_x.push_back((board_width - 1) / 2);
            star_y.push_back(board_height - corner_y - 1);
        }

        if (board_height % 2 != 0 && side_x >= 3 && side_y >= 3 && side_x + side_y >=7)
        {
            star_x.push_back(corner_x);
            star_y.push_back((board_height - 1) / 2);

            star_x.push_back(board_width - corner_x - 1);
            star_y.push_back((board_height - 1) / 2);
        }
    }

    int last_x = -2;
    int last_y = -2;
    if (last_p != NO_MOVE && last_p != PASS_MOVE)
    {
        last_x = posx(last_p);
        last_y = posy(last_p);
    }

    auto B = "\033[1;30;43m";
    auto W = "\033[1;37;43m";
    auto C = "\033[1;35;43m";
    auto G = "\033[0;32m";
    auto BG = "\033[1;;43m";
    auto None = "\033[0m";

    //board
    cerr << "board\n";
    cerr << "    ";
    for (x = 0; x < board_width; ++x)
        cerr << G << (x) % 10 << None << " ";

    cerr << endl;

    for (y = 0; y < board_height; ++y)
    {
        if (y > 9)
            cerr << " ";
        else
            cerr << "  ";
        cerr << G << y << None;
        for (x = 0; x < board_width; ++x)
        {
            if (y == last_y)
            {
                if (x == last_x)
                    cerr << C << "(" << None;
                else if (x == last_x + 1)
                    cerr << C << ")" << None;
                else
                    cerr << BG << " " << None;
            }
            else
                cerr << BG << " " << None;

            if (board[pos(x, y)].color == BLACK)
                cerr << B << "O" << None;
            else if(board[pos(x, y)].color == WHITE)
                cerr << W << "O" << None;
            else
            {
                if (board_width >= 7 && board_height >= 7)
                {
                    bool f = false;
                    for (int s = 0; s < star_x.size(); ++s)
                    {
                        if (x == star_x[s] && y == star_y[s])
                        {
                            cerr << BG << '+' << None;
                            f = true;
                            break;
                        }
                    }

                    if ( !f )
                        cerr << BG << '.' << None;
                }
                else
                    cerr << BG << '.' << None;
            }
        }

        if (y == last_y && last_x == board_width - 1)
            cerr << C << ")" << G << y << None;
        else
            cerr << BG << " " << G << y << None;

        cerr << endl;
    }
    cerr << "    ";
    for (x = 0; x < board_width; ++x)
        cerr << G << (x) % 10 << None << " ";

    cerr << endl;
    cerr << "dead_count: " << dead_count << " dead_black: " << dead_black << " dead_white: " << dead_white << endl;
}

void GoBoard::mv_show(vector<int64_t>& mv, uint64_t N)
{
    int x, y;
    int board_width = xline;
    int board_height = yline;

    int corner_x = (board_width >= 10) ? 3: 2;
    int corner_y = (board_height >= 10) ? 3: 2;

    vector<int> star_x;
    vector<int> star_y;
    if (board_width >= 7 && board_height >= 7)
    {
        star_x.push_back(corner_x);
        star_y.push_back(corner_y);

        star_x.push_back(board_width - corner_x - 1);
        star_y.push_back(corner_y);

        star_x.push_back(corner_x);
        star_y.push_back(board_height - corner_y - 1);

        star_x.push_back(board_width - corner_x - 1);
        star_y.push_back(board_height - corner_y - 1);

        int side_x = (board_width - 1) / 2 - corner_x;
        int side_y = (board_height - 1) / 2 - corner_y;
        if (board_width % 2 != 0 && board_height % 2 != 0 && side_x >= 3 && side_y >= 3)
        {
            star_x.push_back((board_width - 1) / 2);
            star_y.push_back((board_height - 1) / 2);
        }

        if (board_width % 2 != 0 && side_x >= 3 && side_y >= 3 && side_x + side_y >=7)
        {
            star_x.push_back((board_width - 1) / 2);
            star_y.push_back(corner_y);

            star_x.push_back((board_width - 1) / 2);
            star_y.push_back(board_height - corner_y - 1);
        }

        if (board_height % 2 != 0 && side_x >= 3 && side_y >= 3 && side_x + side_y >=7)
        {
            star_x.push_back(corner_x);
            star_y.push_back((board_height - 1) / 2);

            star_x.push_back(board_width - corner_x - 1);
            star_y.push_back((board_height - 1) / 2);
        }
    }

    int last_x = -2;
    int last_y = -2;
    if (last_p != NO_MOVE && last_p != PASS_MOVE)
    {
        last_x = posx(last_p);
        last_y = posy(last_p);
    }

    auto B = "\033[1;30m";
    auto W = "\033[1;37m";
    auto C = "\033[1;35m";
    auto G = "\033[0;32m";
    auto None = "\033[0m";

    //board
    cerr << "MCTS board_value\n";
    cerr << "    ";
    for (x = 0; x < board_width; ++x)
        cerr << G << (x) % 10 << None << " ";

    cerr << endl;

    int64_t mcts_v = 0;
    int64_t dead_threshold = N * 0.33;
    for (y = 0; y < board_height; ++y)
    {
        if (y > 9)
            cerr << " ";
        else
            cerr << "  ";
        cerr << G << y << None;
        for (x = 0; x < board_width; ++x)
        {
            if (y == last_y)
            {
                if (x == last_x)
                    cerr << C << "(" << None;
                else if (x == last_x + 1)
                    cerr << C << ")" << None;
                else
                    cerr << " " << None;
            }
            else
                cerr << " " << None;

            if (board[pos(x, y)].color == BLACK)
            {
                if (mv[x + y * xline] < -dead_threshold)
                    cerr << W << "X" << None;
                else
                    cerr << B << "X" << None;
            }
            else if(board[pos(x, y)].color == WHITE)
            {
                if (mv[x + y * xline] > dead_threshold)
                    cerr << B << "O" << None;
                else
                    cerr << W << "O" << None;
            }
            else
            {
                char e = '.';
                if (board_width >= 7 && board_height >= 7)
                {
                    for (int s = 0; s < star_x.size(); ++s)
                    {
                        if (x == star_x[s] && y == star_y[s])
                        {
                            e = '+';
                            break;
                        }
                    }
                }

                if (mv[x + y * xline] > dead_threshold)
                    cerr << B << e << None;
                else if (mv[x + y * xline] < -dead_threshold)
                    cerr << W << e << None;
                else
                    cerr << e;
            }

            mcts_v += mv[x + y * xline];
        }

        if (y == last_y && last_x == board_width - 1)
            cerr << C << ")" << G << y << None;
        else
            cerr << " " << G << y << None;

        cerr << endl;
    }
    cerr << "    ";
    for (x = 0; x < board_width; ++x)
        cerr << G << (x) % 10 << None << " ";

    cerr << endl;
    cerr << "MCTS board_value -> V: " << float(mcts_v) / N << endl;
}

void GoBoard::update_planes()
{
    vector<float> black(xline * yline, 0.0f);
    vector<float> white(xline * yline, 0.0f);

    auto c = 0;
    for (int y = 0; y < yline; ++y)
    {
        for (int x = 0; x < xline; ++x)
        {
            c = board[pos(x, y)].color;
            if (c == BLACK)
                black[x + y * xline] = 1.0f;
            else if (c == WHITE)
                white[x + y * xline] = 1.0f;
        }
    }
    black_planes.emplace_back(black);
    white_planes.emplace_back(white);

    total_planes.emplace_back(white);
    total_planes.emplace_back(black);
}

void GoBoard::get_planes(vector<float>& stone, int history)
{
    int board_size = xline * yline;
    stone.clear();
    stone.resize(history * 2 * board_size, 0.0f);
    int size = black_planes.size();
    int pl_start = 0;
    int it = 0;
    int skip = 0;
    if (size < history)
        skip = (history - size) * board_size;
    else
        pl_start = size - history;

    if ( next_black() )
    {
        it += skip;
        for (int pl = pl_start; pl < size; ++pl)
        {
            for (auto v: white_planes[pl])
            {
                stone[it] = v;
                ++it;
            }
        }

        it += skip;
        for (int pl = pl_start; pl < size; ++pl)
        {
            for (auto v: black_planes[pl])
            {
                stone[it] = v;
                ++it;
            }
        }
    }
    else
    {
        it += skip;
        for (int pl = pl_start; pl < size; ++pl)
        {
            for (auto v: black_planes[pl])
            {
                stone[it] = v;
                ++it;
            }
        }

        it += skip;
        for (int pl = pl_start; pl < size; ++pl)
        {
            for (auto v: white_planes[pl])
            {
                stone[it] = v;
                ++it;
            }
        }
    }
}

void GoBoard::get_planes01(vector<float>& stone, int history, int mode, int shuffle, int p, int selu)
{
    int board_size = xline * yline;
    stone.clear();
    stone.resize((history * 2 + COLOR_PLANES + ACTION_PLANES) * board_size, 0.0f);
    int size = total_planes.size();
    int pl_start = 0;
    int it = 0;
    int skip = 0;
    if (size < history * 2)
        skip = (history * 2 - size) * board_size;
    else
        pl_start = size - history * 2;

    // 获取旋转后棋盘平面
    if (mode == 1)
    {
        it += skip;
        for (int pl = pl_start; pl < size; ++pl)
        {
            for (auto v: total_planes[pl])
            {
                stone[it] = v;
                ++it;
            }
        }
    }
    else
    {
        it += skip;
        for (int pl = pl_start; pl < size; ++pl)
        {
            auto& plane = total_planes[pl];
            auto* src = plane.data();
            auto* dst = &stone[it];
            int p, new_p;
            for (int y = 0; y < yline; ++y)
            {
                for (int x = 0; x < xline; ++x)
                {
                    p = x + y * xline;
                    new_p = transform_p(p, xline, yline, mode);
                    dst[new_p] = src[p];
                }
            }
            it += plane.size();
        }
    }

    // 颜色平面
    if ( next_black() )
        it += board_size;

    for (auto v: one_plane)
    {
        stone[it] = v;
        ++it;
    }
    
    // 动作平面
    if (ACTION_PLANES > 0)
    {
        if ( !next_black() )
            it += board_size;
        
        if (p == GoBoard::NO_MOVE)
        {
            // state
            if ( ACTION_PLANES > 1)
            {
                // action[0][:] = 0.0
                // action[1][:] = 1.0
                it += board_size;
                for (auto v: one_plane)
                {
                    stone[it] = v;
                    ++it;
                }
            }
            // else
            // action[0][:] = 0.0
        }
        else if (p != GoBoard::PASS_MOVE)
        {
            // normal move: action[0][np] = 1.0
            auto np = pos_no_pad(p);
            stone[it + np] = 1.0f;
        }
        // else 
        // pass move: action[0][:] = 0.0
    }

    // 黑白互换
    if (shuffle)
    {
        vector<float> tmp = stone;
        int planes = history * 2 + COLOR_PLANES;// 动作平面 不用换
        for (int i = 0; i < planes; i += 2)
        {
            for (int k = 0; k < board_size; ++k)
            {
                stone[k + i * board_size] = tmp[k + (i + 1) * board_size];
            }
        }

        for (int i = 0; i < planes; i += 2)
        {
            for (int k = 0; k < board_size; ++k)
            {
                stone[k + (i + 1) * board_size] = tmp[k + i * board_size];
            }
        }
    }

    // selu
    if (selu)
    {
        int planes = history * 2 + COLOR_PLANES + ACTION_PLANES;
        transform_selu(stone.data(), planes, xline, yline);
    }
}

bool GoBoard::swap_action()
{
    if (m_move_count <= 2)
        return false;

    for (auto v: take_history)
    {
        if (v)
            return false;
    }
    return true;
}

// static
vector<int> GoBoard::calc_action(const vector<float>& stone, int history, int size)
{
    vector<int> action;
    for (int i = 1; i < history; ++i)
    {
        auto* prev_white_board = &stone[(i * 2 - 2) * size];
        auto* prev_black_board = &stone[(i * 2 - 1) * size];
        auto* curr_white_board = &stone[(i * 2 + 0) * size];
        auto* curr_black_board = &stone[(i * 2 + 1) * size];
        for (int p = 0; p < size; ++p)
        {
            if (prev_white_board[p] != curr_white_board[p])
            {
                action.push_back(p << 1);
                break;
            }

            if (prev_black_board[p] != curr_black_board[p])
            {
                action.push_back((p << 1) | 1);
                break;
            }
        }
    }

    return action;
}

// static
vector<int> GoBoard::shuffle_action(vector<int>& action)
{
    vector<int> act1;
    vector<int> act2;
    for (int i = 0; i < action.size(); ++i)
    {
        if (i & 1)
            act2.push_back(action[i]);
        else
            act1.push_back(action[i]);
    }

    shuffle(act1.begin(), act1.end(), std::default_random_engine(GoRandom::Get().R32()));
    shuffle(act2.begin(), act2.end(), std::default_random_engine(GoRandom::Get().R32()));
    vector<int> shuffle_action(action.size());
    for (int i = 0; i < act1.size(); ++i)
        shuffle_action[i * 2] = act1[i];

    for (int i = 0; i < act2.size(); ++i)
        shuffle_action[i * 2 + 1] = act2[i];

    return shuffle_action;
}

void GoBoard::get_state(vector<float>& state)
{
    int board_size = xline * yline;
    state.clear();
    state.resize(board_size * 3, 0.0f);
    
    auto c = 0;
    for (int y = 0; y < yline; ++y)
    {
        for (int x = 0; x < xline; ++x)
        {
            c = board[pos(x, y)].color;
            state[(x + y * xline) + c * board_size] = 1.0f;
        }
    }
}
