#ifndef OPENWAKEWORD_HPP
#define OPENWAKEWORD_HPP

#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <atomic>

class openwakeword_detector
{
public:
    void init(std::string path_to_model);

    uint8_t detect_wakeword(std::vector<float> &samples);
    ~openwakeword_detector();
private:    
    std::thread thr;
    std::atomic_bool _wake_word_detected = false;
    std::atomic_bool _do_exit = false;
    std::vector<float> m_float_samples;
    std::atomic_bool m_new_samples = false;
};

#endif /* OPENWAKEWORD_HPP */
