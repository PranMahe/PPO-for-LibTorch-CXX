#pragma once
#include "PPOAgent.h"
#include "environments/CartPoleEnv.h"


template<typename EnvType>
void PPOAgent::collectRollouts(EnvType& env) {
    // Clear any experience from previous rollout
    buffer.clear();

    // Collect num_envs episodes of experience
    for (int ep = 0; ep < config.num_envs; ep++) {
        
        // Reset Environment
        std::vector<float> obs = env.reset();

        // Run one Episode
        for (int t = 0; t < config.steps_per_ep; t++) {
            torch::Tensor obsTensor = obsToTensor(obs);

            if (config.action_type == "discrete") {
                auto [action, logProb, value] = network->selectAction(obsTensor);
                StepResult result = env.step(action);

                buffer.observations.push_back(obsTensor);
                buffer.actions.push_back(action);
                buffer.logProbs.push_back(logProb);
                buffer.values.push_back(value);
                buffer.rewards.push_back(result.reward);
                buffer.dones.push_back(result.done);

                obs = result.observation;
                if (result.done){
                    break;
                }
            }
            else {
                auto [actionVec, logProb, value] = network->selectActionContinuous(obsTensor);
                StepResult result = env.stepContinuous(actionVec);

                buffer.observations.push_back(obsTensor);
                buffer.continuousActions.push_back(actionVec);
                buffer.logProbs.push_back(logProb);
                buffer.values.push_back(value);
                buffer.rewards.push_back(result.reward);
                buffer.dones.push_back(result.done);

                obs = result.observation;
                if (result.done){
                    break;
                }
            }
        }
    }
}