//
// Created by yuanyu on 2018.01.24.
//

#pragma once

#include <iostream>
#include <chrono>
#include <string>

class StopWatch
{
public:
    StopWatch() = default;
    void start()
    {
        time_start = std::chrono::steady_clock::now();
    }

    void end(const std::string& hint = "time")
    {
        auto time_end = std::chrono::steady_clock::now();
        auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_start).count();
        std::cerr << hint << ": " << time_used << " seconds." << std::endl;
    }

    bool timeout(double second)
    { 
        auto time_end = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_start).count() >= second;
    }

    void start_count()
    {
        count_time_start = std::chrono::steady_clock::now();
    }

    void end_count()
    {
        auto time_end = std::chrono::steady_clock::now();
        count += std::chrono::duration_cast<std::chrono::duration<double>>(time_end - count_time_start).count();
    }

    void output_count(const std::string& hint = "time")
    {
        std::cerr << hint << ": " << count << " seconds." << std::endl;
    }

    void clear_count()
    {
        count = 0.0;
    }

    bool timeout_count(double second)
    { 
        auto time_end = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double>>(time_end - count_time_start).count() >= second;
    }

private:
    std::chrono::steady_clock::time_point time_start, count_time_start;
    double count = 0.0;
};

int mkpath(std::string s, mode_t mode = 0755);
void folder_init(int width, int height);
int next_file_id(std::string fn);
int get_file_id(std::string fn);
int get_file_id(int width, int height);
std::string get_model_folder();
std::string get_data_folder(int width, int height);
std::string get_rd_folder(int width, int height);
std::string get_sgf_folder(int width, int height);