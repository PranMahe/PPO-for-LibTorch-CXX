#include "ActorCritic.h"
#include <torch/torch.h>
#include <cmath>
#include <stdexcept>

// Constructor
ActorCriticImpl::ActorCriticImpl(int stateDim, int hiddenDim, int actionDim_, const std::string& actionType_) : actionType(actionType_), actionDim(actionDim_) {
    
    // Register Layers
    fc1 = register_module("fc1", torch::nn::Linear(stateDim, hiddenDim));
    fc2 = register_module("fc2", torch::nn::Linear(hiddenDim, hiddenDim));

    // Register output heads
    actorHead = register_module("actorHead", torch::nn::Linear(hiddenDim, actionDim));
    criticHead = register_module("criticHead", torch::nn::Linear(hiddenDim, 1));

    // Weight Initialization - Orthogonal
    float gainShared = std::sqrt(2.0f);

    torch::nn::init::orthogonal_(fc1->weight, gainShared);
    torch::nn::init::constant_(fc1->bias, 0.0f);

    torch::nn::init::orthogonal_(fc2->weight, gainShared);
    torch::nn::init::constant_(fc2->bias, 0.0f);

    torch::nn::init::orthogonal_(actorHead->weight, 0.01f);
    torch::nn::init::constant_(actorHead->bias, 0.0f);

    torch::nn::init::orthogonal_(criticHead->weight, 0.01f);
    torch::nn::init::constant_(criticHead->bias, 0.0f);

    // Continuous Mode
    if (actionType == "continuous") {
        logStd = register_parameter(
            "logStd",
            torch::zeros({actionDim})
        );
    }
}

// forward()
std::pair<torch::Tensor, torch::Tensor> ActorCriticImpl::forward(torch::Tensor input) {
    // Forward call the layer
    torch::Tensor x = torch::tanh(fc1->forward(input));
    x = torch::tanh(fc2->forward(x));

    // Actor
    torch::Tensor logits = actorHead->forward(x);

    // Critic
    torch::Tensor value = criticHead->forward(x).squeeze(-1);

    return {logits, value};
}

// selectAction() - Discrete
std::tuple<int, float, float> ActorCriticImpl::selectAction(torch::Tensor obs){
    torch::NoGradGuard noGrad;

    // Run forward pass
    auto [logits, value] = forward(obs);

    // Sample action from Categorical
    torch::Tensor probs = torch::softmax(logits, -1);

    // Sample action using multinomial
    torch::Tensor actionTensor = torch::multinomial(probs, 1).squeeze(-1);
    
    // Log Probability
    torch::Tensor logProbTensor = torch::log(
        probs.gather(-1, actionTensor.unsqueeze(-1))
    ).squeeze(-1);

    // Extract scalar vals
    int action = actionTensor.item<int>();
    float logProb = logProbTensor.item<float>();
    float val = value.item<float>();

    return {action, logProb, val};
}

//selectActionContinuous()
std::tuple<std::vector<float>, float, float>ActorCriticImpl::selectActionContinuous(torch::Tensor obs) {
    torch::NoGradGuard noGrad;
    auto [mean, value] = forward(obs);

    torch::Tensor clampedLogStd = logStd.clamp(-2.0f, 2.0f);
    torch::Tensor std = torch::exp(clampedLogStd);

    // Reparam trick
    torch::Tensor noise = torch::randn_like(mean);
    torch::Tensor sample = mean + std * noise;

    // MountainCarContinuous Range [-1, 1]
    torch::Tensor action = torch::tanh(sample);

    // Log Prob Gaussian
    // log p(a) = log N(u | mean, std) - sum(log (1 - tanh(u)^2))
    torch::Tensor logProbGaussian = 
        -0.5f * torch::pow((sample - mean) / (std + 1e-8f), 2)
        - torch::log(std + 1e-8f)
        -0.5f * std::log(2.0f * M_PI);

    torch::Tensor squashCorrection = torch::log(1.0f - action.pow(2) + 1e-6f);
    torch::Tensor logProb = (logProbGaussian - squashCorrection).sum(-1);

    std::vector<float> actionVec(actionDim);
    for (int i = 0; i < actionDim; i++) {
        actionVec[i] = action[0][i].item<float>();
    }

    return {actionVec, logProb.item<float>(), value.item<float>()};
}

// getValue()
float ActorCriticImpl::getValue(torch::Tensor obs) {
    torch::NoGradGuard noGrad;

    auto [logits, value] = forward(obs);
    std::ignore = logits;
    
    return value.item<float>();
}

// computeLogProbs()
torch::Tensor ActorCriticImpl::computeLogProbs(const torch::Tensor& actorOutput, const torch::Tensor& actions) {
    if (actionType == "discrete") {
        torch::Tensor probs =  torch::softmax(actorOutput, -1);
        torch::Tensor logProbs = torch::log(probs + 1e-8f);
        
        return logProbs.gather(-1, actions.unsqueeze(-1)).squeeze(-1);
    }
    else {
        torch::Tensor aClamped = actions.clamp(-1.0f + 1e-6f, 1.0f - 1e-6f);
        torch::Tensor u = torch::atanh(aClamped);

        torch::Tensor clampedLogStd = logStd.clamp(-2.0f, 2.0f);
        torch::Tensor std = torch::exp(clampedLogStd);

        torch::Tensor logProbGaussian = 
            -0.5f * torch::pow((u - actorOutput) / (std + 1e-8f), 2)
            - torch::log(std + 1e-8f)
            -0.5f * std::log(2.0f * M_PI);

        torch::Tensor squashCorrection = torch::log(1.0f - actions.pow(2) + 1e-6f);
        return (logProbGaussian - squashCorrection).sum(-1);
    }
}

// computeEntropy()
torch::Tensor ActorCriticImpl::computeEntropy(const torch::Tensor& actorOutput) {
    if (actionType == "discrete") {
        torch::Tensor probs = torch::softmax(actorOutput, -1);
        torch::Tensor logProbs = torch::log(probs + 1e-8f);

        return -(probs * logProbs).sum(-1).mean();
    }
    else {
        // Gaussian Entropy
        torch::Tensor clampedLogStd = logStd.clamp(-2.0f, 2.0f);
        return (clampedLogStd + 0.5f * std::log(2.0f * M_PI * M_E)).sum().mean();
    }
}