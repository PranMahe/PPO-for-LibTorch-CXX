#pragma once

#include <torch/torch.h>
#include <string.h>

// ActorCriticImpl

struct ActorCriticImpl : torch::nn::Module {
    // Layers
    torch::nn::Linear fc1{nullptr}; // input -> hidden
    torch::nn::Linear fc2{nullptr}; // input -> hidden

    // Actor Head
    torch::nn::Linear actorHead{nullptr};

    // Critic Head
    torch::nn::Linear criticHead{nullptr};

    // Continuous Actions: Log Standard
    torch::Tensor logStd;

    // Policy Type
    std::string actionType;
    int actionDim;

    // Constructor
    ActorCriticImpl(int stateDim, int hiddenDim, int actionDim, const std::string& actionType = "discrete");

    // Forward Pass
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

    // Discrete Action Selection
    std::tuple<int, float, float> selectAction(torch::Tensor obs);

    // Continuous Action Selection
    std::tuple<std::vector<float>, float, float> selectActionContinuous(torch::Tensor obs);

    // Value Estimate
    float getValue(torch::Tensor obs);

    // Log Probability
    torch::Tensor computeLogProbs(const torch::Tensor& actorOutput, const torch::Tensor& actions);

    // Entropy
    torch::Tensor computeEntropy(const torch::Tensor& actorOutput);
};

// LibTorch shared ptr to ActorCriticImpl
TORCH_MODULE(ActorCritic);