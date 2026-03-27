#pragma once

#include "networks/ActorCritic.h"
#include "config/Config.h"

#include <vector>
#include <memory>
#include <string>
#include <torch/torch.h>

// RolloutBuffer
struct RolloutBuffer {
    std::vector<torch::Tensor> observations;
    std::vector<int> actions;
    std::vector<std::vector<float>> continuousActions;
    std::vector<float> logProbs;
    std::vector<float> values;
    std::vector<float> rewards;
    std::vector<bool> dones;

    // Clear all stored data
    void clear() {
        observations.clear();
        actions.clear();
        continuousActions.clear();
        logProbs.clear();
        values.clear();
        rewards.clear();
        dones.clear();
    }

    // Return num stored timesteps
    int size() const {
        return static_cast<int>(rewards.size());
    }
};

// PPOAgent
class PPOAgent {
public:
    explicit PPOAgent(const Config& config);
    
    // Training Interface
    template<typename EnvType>
    void collectRollouts(EnvType& env);

    // PPO Update
    float update();

    // Utilities
    
    // Saves model weight to file
    void save(const std::string& path) const;

    // Loads model weights from a file
    void load(const std::string& path);

    // Returns the current network
    ActorCritic getNetwork() const;

private:
    // Core Components
    ActorCritic network;
    std:: unique_ptr<torch::optim::Adam> actorOptimizer;
    std:: unique_ptr<torch::optim::Adam> criticOptimizer;
    RolloutBuffer buffer;
    Config config;
    torch::Device device;

    // Internal Methods
    // Compute GAE Advantages
    std::pair<torch::Tensor, torch::Tensor> computeGAE();

    // Vector to tensor conversion for obs
    torch::Tensor obsToTensor(const std::vector<float>& obs) const;

    // Run an Epoch
    float runUpdateEpoch(
        const torch::Tensor& obsBatch,
        const torch::Tensor& actionBatch,
        const torch::Tensor& logProbBatch,
        const torch::Tensor& advantageBatch,
        const torch::Tensor& returnBatch
    );
};

#include "core/PPOAgent.tpp"