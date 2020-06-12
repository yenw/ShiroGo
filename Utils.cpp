//
// Created by yuanyu on 2018.01.24.
//

#include <thread>
#include <iostream>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <cerrno>
#include "Utils.h"
#include "config.h"

int mkpath(std::string s, mode_t mode)
{
    size_t pre = 0;
    size_t pos;
    std::string dir;
    int mdret;

    if (s[s.size()-1] != '/')
        s +='/';

    while (true)
    {
        pos = s.find_first_of('/', pre);
        if (pos == std::string::npos)
            break;

        dir = s.substr(0, ++pos);
        pre = pos;

        if ( dir.empty() )
            continue;

        mdret = ::mkdir(dir.c_str(), mode);
        if (mdret && errno != EEXIST)
        {
            return mdret;
        }
    }
    return mdret;
}

int next_file_id(std::string fn)
{
    FILE *fp = fopen(fn.c_str(), "r+");
    if (fp == nullptr)
    {
        fp = fopen(fn.c_str(), "w+");
        if (fp == nullptr)
            return -1;
    }

    auto next_id = 0;
    auto fd = fileno(fp);
    flock(fd, LOCK_EX); //文件加锁

    fseek(fp, 0L, SEEK_END);
    auto size = ftell(fp);
    if ( size != 0)
    {
        fseek(fp, 0L, SEEK_SET);
        char buf[64] = {0};
        fread(buf, 63, 1, fp);
        next_id = atoi(buf);
    }

    ++next_id;
    std::string s_id = std::to_string(next_id);
    freopen(fn.c_str(), "w+", fp);
    fwrite(s_id.c_str(), s_id.size(), 1, fp);
    fclose(fp); //关闭文件
    flock(fd, LOCK_UN); //释放文件锁
    return next_id;
}

int get_file_id(std::string fn)
{
    FILE *fp = fopen(fn.c_str(), "r+");
    if (fp == nullptr)
    {
        fp = fopen(fn.c_str(), "w+");
        if (fp == nullptr)
            return -1;
    }

    auto next_id = 0;
    auto fd = fileno(fp);
    flock(fd, LOCK_EX); //文件加锁

    fseek(fp, 0L, SEEK_END);
    auto size = ftell(fp);
    if ( size != 0)
    {
        fseek(fp, 0L, SEEK_SET);
        char buf[64] = {0};
        fread(buf, 63, 1, fp);
        next_id = atoi(buf);
    }

    fclose(fp); //关闭文件
    flock(fd, LOCK_UN); //释放文件锁
    return next_id;
}

int get_file_id(int width, int height)
{
    return get_file_id(get_data_folder(width, height) + "/id.txt");
}

void folder_init(int width, int height)
{
    mkpath(get_model_folder());
    mkpath(get_sgf_folder(width, height));
    mkpath(get_data_folder(width, height));
    mkpath(get_rd_folder(width, height));
}

std::string get_model_folder()
{
    return "./" + FOLDER_MODEL;
}

std::string get_data_folder(int width, int height)
{
    return "./" + FOLDER_DATA + "/" + std::to_string(height) + 'x' + std::to_string(width);
}

std::string get_rd_folder(int width, int height)
{
    return "./" + FOLDER_RD + "/" + std::to_string(height) + 'x' + std::to_string(width);
}

std::string get_sgf_folder(int width, int height)
{
    return "./" + FOLDER_SGF + "/" + std::to_string(height) + 'x' + std::to_string(width);
}