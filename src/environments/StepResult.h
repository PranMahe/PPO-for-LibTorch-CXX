#pragma once

#include <vector>

struct StepResult {
    std::vector<float> observation;
    float reward;
    bool done;
};