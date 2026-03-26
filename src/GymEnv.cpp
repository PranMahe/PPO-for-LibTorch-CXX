#include "GymEnv.h"

#include <stdexcept>
#include <iostream>

// Constructor
GymEnv::GymEnv(const std::string& envName, int seed):done(false){
    try{
        //import gymnasium as gym
        py::object gym = py::module_::import("gymnasium");

        // env = gym.make(envName)
        env = gym.attr("make")(envName);
        
        // get action space
        try {
            actionDim = env.attr("action_space").attr("n").cast<int>();
        } 
        catch (...) {
            // Continuous
            actionDim = env.attr("action_space")
                            .attr("shape")
                            .attr("__getitem__")(0)
                            .cast<int>(); 
        }
        
        
        // get observation space size
        observationDim = env.attr("observation_space")
                            .attr("shape")
                            .attr("__getitem__")(0)
                            .cast<int>();
        
        // Seed the env for reproducibility
        if (seed != -1) {
            env.attr("reset")(py::arg("seed") = seed);
        }
        
        std::cout   << "GymEnv created: " << envName
                    << " | state_dim=" << observationDim
                    << " | action_dim=" << actionDim
                    << std::endl;
    }
    // Python Traceback error
    catch (py::error_already_set& e) {
        throw std::runtime_error(
            std::string("Failed to create Gymnasium environment: ") + e.what()
        );
    }
}

// reset()
std::vector<float> GymEnv::reset() {
    try {
        // env.reset() returns a tuple: (observation, info)
        py::tuple result =  env.attr("reset")();

        // Extract just the observation
        py::object obs = result[0];

        done = false;
        return extractObservation(obs);
    }
    catch (py::error_already_set& e) {
        throw std::runtime_error (
            std::string("GymEnv::reset() failed: ") + e.what()
        );
    }
}

// step()
StepResult GymEnv::step(int action) {
    if (done) {
        throw std::runtime_error(
            "Cannot step in a finished episode. Call reset() first."
        );
    }

    try {
        // env.step() returns a 5-tuple: (observation, reward, terminated, truncated, info)
        py::tuple result = env.attr("step")(action);

        // Extract each element
        py::object obs = result[0];
        float reward = result[1].cast<float>();
        bool terminated = result[2].cast<bool>();
        bool truncated = result[3].cast<bool>();
        // info not needed

        // episode is done if either terminated or truncated
        done = terminated || truncated;

        // Build and return result
        StepResult stepResult;
        stepResult.observation = extractObservation(obs);
        stepResult.reward = reward;
        stepResult.done = done;
        return stepResult;
    }
    catch (py::error_already_set& e){
        throw std::runtime_error(
            std::string("GymEnv::step() failed: ") + e.what()
        );
    }
}

// stepContinuous()
StepResult GymEnv::stepContinuous(const std::vector<float>& actionVec) {
    if (done){
        throw std::runtime_error(
            "Cannot step in a finished episode. Call reset() first."
        );
    }
    try {
        py::list pyAction;
        for (float v : actionVec) pyAction.append(v);

        py::tuple result = env.attr("step")(pyAction);

        bool terminated = result[2].cast<bool>();
        bool truncated = result[3].cast<bool>();
        done = terminated || truncated;

        StepResult r;
        r.observation = extractObservation(result[0]);
        r.reward = result[1].cast<float>();
        r.done = done;
        return r;
    }
    catch (py::error_already_set& e) {
        throw std::runtime_error(
            std::string("GymEnv::stepContinuous() failed: ") + e.what()
        );
    }
}

// extractObservation()
std::vector<float> GymEnv::extractObservation(py::object obs){
    return obs.cast<std::vector<float>>();
}

// Accessors
int GymEnv::getActionDim() const {
    return actionDim;
}

int GymEnv::getObservationDim() const {
    return observationDim;
}

bool GymEnv::isDone() const {
    return done;
}