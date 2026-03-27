#pragma once

#include "environments/StepResult.h"
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Gymnasium Environment wrapper
class GymEnv{
public:
    // Constructor
    explicit GymEnv(const std::string& envName, int seed = -1);
    // Destructor
    ~GymEnv() = default;

    std::vector<float> reset();
    StepResult step(int action);
    StepResult stepContinuous(const std::vector<float>& actionVec);

    // Accessors
    int getActionDim()      const;
    int getObservationDim() const;
    bool isDone()           const;

private:
    py::object  env;
    bool        done;
    int         actionDim;
    int         observationDim;

    std::vector<float> extractObservation(py::object obs);
};