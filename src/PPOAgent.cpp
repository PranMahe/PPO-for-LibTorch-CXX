#include "PPOAgent.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>

// Constructor
PPOAgent::PPOAgent(const Config& config):
    config(config),
    // Auto use GPU, otherwise CPU
    device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
    // Create network dimensions
    network(config.state_dim, config.hidden_dim, config.action_dim, config.action_type)
{
    // move network to correct device
    network->to(device);

    // Actor Params
    std::vector<torch::Tensor> actorParams;
    for (auto& p : network->fc1->parameters()){ 
        actorParams.push_back(p);
    }
    for (auto& p : network->fc2->parameters()){ 
        actorParams.push_back(p);
    }
    for (auto& p : network->actorHead->parameters()){ 
        actorParams.push_back(p);
    }
    if (config.action_type == "continuous") {
        actorParams.push_back(network->logStd);
    }

    // Critic Params
    std::vector<torch::Tensor> criticParams;
    for (auto& p : network->criticHead->parameters()){ 
        criticParams.push_back(p);
    }

    // Actor Optimizer
    actorOptimizer = std::make_unique<torch::optim::Adam>(
        actorParams,
        torch::optim::AdamOptions(config.actor_lr)
    );

    // Critic Optimizer
    criticOptimizer = std::make_unique<torch::optim::Adam>(
        criticParams,
        torch::optim::AdamOptions(config.critic_lr)
    );
    std::cout   << "PPOAgent Initialized on: "
                << (torch::cuda::is_available() ? "CUDA (GPU)" : "CPU")
                << std::endl;
}

// obsToTensor()
// converts vector<float> observation to tensor
torch::Tensor PPOAgent::obsToTensor(const std::vector<float>& obs) const {
    return torch::tensor(
        obs,
        torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device)
    ).unsqueeze(0);
}

// ComputeGAE()
// Computes Generalized Advantage Estimation working backwards through the rollout buffer
// Returns advantages, discounted_rewards as normalized tensors
std::pair<torch::Tensor, torch::Tensor> PPOAgent::computeGAE() {
    int n = buffer.size();

    std::vector<float> advantages (n, 0.0f);
    std::vector<float> returns(n, 0.0f);

    float lastAdvantage = 0.0f;

    // Work backwards through the buffer
    for (int t = n - 1; t >= 0; t--) {
        // if step ends episode, next state value is 0
        float nextValue  = 0.0f;
        if (t < n - 1 && !buffer.dones[t]) {
            nextValue = buffer.values[t + 1];
        }

        // TD error
        float delta = buffer.rewards[t] + config.gamma * nextValue - buffer.values[t];

        // GAE Advantage - combines current delta with future advantages
        float doneMask = buffer.dones[t] ? 0.0f : 1.0f;
        lastAdvantage = delta + config.gamma * config.gae_lambda * doneMask * lastAdvantage;

        advantages[t] = lastAdvantage;
        returns[t] = advantages[t] + buffer.values[t];
    }
    // convert to tensor
    torch::Tensor advTensor = torch::tensor(
        advantages, torch::TensorOptions().dtype(torch::kFloat32).device(device)
    );

    torch::Tensor retTensor = torch::tensor(
        returns, torch::TensorOptions().dtype(torch::kFloat32).device(device)
    );

    // Normalize advantages
    float eps = 1e-8f;
    advTensor = (advTensor - advTensor.mean()) / (advTensor.std() + eps);

    return {advTensor, retTensor};
}

float PPOAgent::update() {
    // Stack buffer into tensors
    torch::Tensor obsBatch = torch::cat(buffer.observations, 0);

    // Convert vectors to Tensors    
    torch::Tensor logProbBatch = torch::tensor(
        buffer.logProbs, torch::TensorOptions().dtype(torch::kFloat32).device(device)
    );

    torch::Tensor actionBatch;
    // Discrete
    if (config.action_type == "discrete") {
        actionBatch = torch::tensor(
            buffer.actions, torch::TensorOptions().dtype(torch::kInt64).device(device)
        );
    }
    // Continuous
    else {
        int n = buffer.size();
        std::vector<float> flat;
        flat.reserve(n* config.action_dim);
        for (auto& a : buffer.continuousActions){
            flat.insert(flat.end(), a.begin(), a.end());
        }

        actionBatch = torch::tensor(
            flat,
            torch::TensorOptions().dtype(torch::kFloat32).device(device)
        ).view({n, config.action_dim});
    }

    // Compute GAE Advantages and Discounted Returns
    auto [advantages, returns] = computeGAE();

    // Run PPO Epochs
    float totalLoss = 0.0f;
    
    for (int epoch = 0; epoch < config.epochs; epoch++) {
        totalLoss += runUpdateEpoch(obsBatch, actionBatch, logProbBatch, advantages, returns);
    }

    return totalLoss / config.epochs;
}

// runUpdateEpoch()
float PPOAgent::runUpdateEpoch(
    const torch::Tensor& obsBatch,
    const torch::Tensor& actionBatch,
    const torch::Tensor& logProbBatch,
    const torch::Tensor& advantageBatch,
    const torch::Tensor& returnBatch)
{
    int n = buffer.size();
    float epochLoss = 0.0f;
    int numBatches = 0;

    // Shuffle Indices
    std:: vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));

    // Mini-batch Loop
    for (int start = 0; start + config.batch_size <= n; start += config.batch_size) {
        
        // Extract mini-batch indices
        std::vector<int> batchIdx(indices.begin() + start, indices.begin() + start + config.batch_size);

        // Convert to Tensor for indexing
        torch::Tensor idxTensor = torch::tensor(batchIdx, torch::TensorOptions().dtype(torch::kInt64).device(device));

        // Index into full batch to get mini-batch
        torch::Tensor obsMiniBatch = obsBatch.index_select(0, idxTensor);
        torch::Tensor actionMiniBatch = actionBatch.index_select(0, idxTensor);
        torch::Tensor oldLogProbMiniBatch = logProbBatch.index_select(0, idxTensor);
        torch::Tensor advMiniBatch = advantageBatch.index_select(0, idxTensor);
        torch::Tensor retMiniBatch = returnBatch.index_select(0, idxTensor);

        // Forward Pass
        auto [logits, actor_values] = network->forward(obsMiniBatch);

        torch::Tensor newLogProbs = network->computeLogProbs(logits, actionMiniBatch);
        torch::Tensor entropy = network->computeEntropy(logits);

        // PPO Clipped Loss
        torch::Tensor ratio = torch::exp(newLogProbs - oldLogProbMiniBatch);

        // Inclipped Objective
        torch::Tensor surr1 = ratio * advMiniBatch;

        // Clipped Objective
        torch::Tensor surr2 = torch::clamp(
            ratio, 1.0f - config.clip_eps, 1.0f + config.clip_eps
        ) * advMiniBatch;

        // Policy Loss
        torch::Tensor policyLoss = -torch::min(surr1, surr2).mean();

        // Actor Loss
        torch::Tensor actorLoss = policyLoss - config.entropy_coef * entropy;

        // Actor Backward
        actorOptimizer->zero_grad();
        actorLoss.backward();
        torch::nn::utils::clip_grad_norm_(network->fc1->parameters(), 0.5f);
        torch::nn::utils::clip_grad_norm_(network->fc2->parameters(), 0.5f);
        torch::nn::utils::clip_grad_norm_(network->actorHead->parameters(), 0.5f);
        actorOptimizer->step();

        // Critic Backward
        auto[critic_logits, values] = network->forward(obsMiniBatch);
        std::ignore = critic_logits;
        torch::Tensor criticLoss = config.value_coef * torch::mse_loss(values, retMiniBatch);

        criticOptimizer->zero_grad();
        criticLoss.backward();
        torch::nn::utils::clip_grad_norm_(network->criticHead->parameters(), 0.5f);
        criticOptimizer->step();

        epochLoss += (actorLoss + criticLoss).item<float>();
        numBatches++;
    }

    return numBatches > 0 ? epochLoss / numBatches : 0.0f;
}

// save() and load()

void PPOAgent::save(const std::string& path) const {
    torch::serialize::OutputArchive archive;
    network->save(archive);
    archive.save_to(path);
    std::cout << "Model saved to: " << path << std::endl;
}

void PPOAgent::load(const std::string& path) {
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    network->load(archive);
    std::cout <<"Model loaded from: " << path << std::endl;
}

ActorCritic PPOAgent::getNetwork() const {
    return network;
}