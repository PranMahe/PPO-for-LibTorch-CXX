#include <pybind11/embed.h>
#include <iostream>
#include <chrono>
#include <string>
#include <iomanip>
#include <torch/torch.h>

#include "Config.h"
#include "CartPoleEnv.h"
#include "GymEnv.h"
#include "PPOAgent.h"

namespace py = pybind11;

// evaluate()
template <typename EnvType>
float evaluate(PPOAgent& agent, EnvType& env, int numTrials, const Config& config) {
    float totalReward = 0.0f;
    ActorCritic network = agent.getNetwork();

    for (int trial = 0; trial < numTrials; trial++) {
        std::vector<float> obs = env.reset();
        float episodeReward =  0.0f;

        for (int t = 0; t < 1000; t++) {
            //  Convert obs to tensor
            torch::Tensor obsTensor = torch::tensor(
                obs, torch::TensorOptions().dtype(torch::kFloat32)
                .device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
            ).unsqueeze(0);
            
            // Select Action
            torch::NoGradGuard noGrad;
            StepResult result;
            if (config.action_type == "discrete") {
                auto [action, logProb, value] = network->selectAction(obsTensor);
                std::ignore = logProb
                std::ignore = value
                result = env.step(action);
            }
            else {
                auto [actionVec, logProb, value] = network->selectActionContinuous(obsTensor);
                std::ignore = logProb
                std::ignore = value
                result = env.stepContinuous(actionVec);
            }

            // Step Environment
            episodeReward += result.reward;
            obs = result.observation;

            if (result.done) break;
        }

        totalReward += episodeReward;
    }

    return totalReward / numTrials;
}

// runTraining()
template<typename EnvType>
void runTraining(PPOAgent& agent, EnvType& env, const Config& config) {
    std::cout   << "\n=== Starting Training ===" << std::endl;
    std::cout   << "Environment:  " << config.env << std::endl;
    std::cout   << "Max Episodes: " << config.max_episodes << std::endl;
    std::cout   << "Device:       "
                << (torch::cuda::is_available() ? "CUDA (GPU)" : "CPU")
                << "\n" << std::endl;

    float bestReward = -1e9f;
    auto trainStart = std::chrono::high_resolution_clock::now();

    for (int episode = 0; episode < config.max_episodes; episode++) {
        auto iterStart = std::chrono::high_resolution_clock::now();

        // Collect Experience
        agent.collectRollouts(env);

        // Update Policy
        float loss = agent.update();

        // Timing
        auto iterEnd = std::chrono::high_resolution_clock::now();
        auto iterMs = std::chrono::duration_cast<std::chrono::milliseconds>(iterEnd - iterStart).count();

        // Periodic Evaluation
        if (episode % config.test_interval == 0) {
            float avgReward = evaluate(agent, env, config.test_trials, config);

            // Formatted Output
            std::cout   << "Episode: " << std::setw(5) << episode
                        << " | Reward: " << std::setw(8)
                        << std::fixed << std::setprecision(2) << avgReward
                        << " | Loss: " << std::setw(8)
                        << std::fixed << std::setprecision(2) << loss
                        << " | Time: " << iterMs << "ms"
                        << std::endl;

            // Save Best Model
            if (avgReward > bestReward) {
                bestReward = avgReward;
                agent.save("best_model.pt");
            }
        }

        // Periodic Checkpoint
        if (episode % config.save_interval == 0 && episode > 0) {
            std::string checkpointPath = "checkpoint_ep" + std::to_string(episode) + ".pt";
            agent.save(checkpointPath);
        }
    }

    // Training Complete
    auto trainEnd = std::chrono::high_resolution_clock::now();
    auto trainSec = std::chrono::duration_cast<std::chrono::seconds>(trainEnd - trainStart).count();

    std::cout   <<"\n=== Training Complete ===" << std::endl;
    std::cout   << "Total Time: " << trainSec << "s" << std::endl;
    std::cout   << "Best reward: " << bestReward << std::endl;

    // Final Evaluation
    float finalReward = evaluate(agent, env, 20, config);
    std::cout << "Final Reward (avg over 20 trials): " << finalReward << std::endl;
}

// main()

int main() {
    try {
        // Config
        Config config = Config::fromFile("config.json");

        // Python Interpreter
        py::scoped_interpreter guard{};

        // Agent
        PPOAgent agent(config);

        // Environment and Training
        if (config.env == "cartpole") {
            CartPoleEnv env;
            runTraining(agent, env, config);
        }
        else if (config.env == "lunarlander") {
            GymEnv env("LunarLander-v3");
            runTraining(agent, env, config);
        }
        else if (config.env == "mountaincar") {
            GymEnv env("MountainCarContinuous-v0");
            runTraining(agent, env, config);
        }
        else {
            throw std::invalid_argument(
                "Unknown Environment: " + config.env
            );
        }

        return 0;

    }catch (const std::exception& e) {
        std::cerr << "\nFatal error: " << e.what() << std::endl;
        return 1;
    }
}