//
// Created by yuanyu on 2018.01.30.
//

#include "Transform.h"

void transform_xy(int& x, int& y, int width, int height, int mode)
{
    int val_x;
    int val_y;
    switch (mode)
    {
        case 2:
        {
            val_x = height - 1 - y;
            val_y = width - 1 - x;
            break;
        }
        case 3:
        {
            val_y = width - 1 - x;
            val_x = y;
            break;
        }
        case 4:
        {
            val_x = width - 1 - x;
            val_y = y;
            break;
        }
        case 5:
        {
            val_x = width - 1 - x;
            val_y = height - 1 - y;
            break;
        }
        case 6:
        {
            val_x = y;
            val_y = x;
            break;
        }
        case 7:
        {
            val_x = height - 1 - y;
            val_y = x;
            break;
        }
        case 8:
        {
            val_y = height - 1 - y;
            val_x = x;
            break;
        }
        default:
            val_x = x;
            val_y = y;
    }
    x = val_x;
    y = val_y;
}

int transform_p(int p, int width, int height, int mode)
{
    auto new_x = p % width;
    auto new_y = p / width;
    transform_xy(new_x, new_y, width, height, mode);
    return new_x + new_y * width;
}