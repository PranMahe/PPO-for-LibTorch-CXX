#pragma once

#include <string>
#include <stdexcept>
#include <fstream>
#include <nlohmann/json.hpp>

struct Config {
   
    // Environment Dimensions
    std::string env;
    int state_dim;
    int action_dim;
    std::string action_type;    // Discrete or Continuous

    // Network Architecture
    int hidden_dim;

    // PPO Hyperparameters
    float gamma;            // Discount Factor
    float gae_lambda;       // GAE Smoothing
    float clip_eps;         // PPO clip range
    float entropy_coef;     // Encourages exploration
    float value_coef;       // Critic loss contribution

    // Training
    int max_episodes;       // Total number of training episodes
    int steps_per_ep;       // Max timesteps per episode
    float actor_lr;         // Learning rate for actor parameters
    float critic_lr;        // Learning rate for critic parameters
    int epochs;             // Batch experience reuse
    int batch_size;         // Number of samples per gradient update
    int num_envs;           // Episodes to collect before each update
    
    // Logging
    int test_interval;      // Test every N episodes
    int test_trials;        // Average over N test runs
    int save_interval;      // Save checkpoint every N episodes

    // Default Constructor
    Config() = default;
    // Environment constructor
    explicit Config(const std::string& envName) : env(envName) {
        setEnvDimensions();
    }

    // JSON Config Loading
    static Config fromFile (const std::string& path) {
        // Open the file
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error(
                "Could not open config file: " + path + "\nMake sure config.json exists next to executable."
            );
        }

        // Parse JSON
        nlohmann::json j;
        file >> j;

        // Build Config
        Config config;
        config.env = j.value("env", "lunarlander");

        if (j.contains("network")) {
            config.hidden_dim = j["network"].value("hidden_dim", 128);
        }
        if (j.contains("ppo")) {
            config.gamma = j["ppo"].value("gamma", 0.99f);
            config.gae_lambda = j["ppo"].value("gae_lambda", 0.95f);
            config.clip_eps = j["ppo"].value("clip_eps", 0.2f);
            config.entropy_coef = j["ppo"].value("entropy_coef", 0.01f);
            config.value_coef = j["ppo"].value("value_coef", 0.5f);
        }
        if (j.contains("training")) {
            float sharedLr = j["training"].value("lr", 3e-4f);
            config.actor_lr = j["training"].value("actor_lr", sharedLr);
            config.critic_lr = j["training"].value("critic_lr", sharedLr);
            config.epochs = j["training"].value("epochs", 10);
            config.batch_size = j["training"].value("batch_size", 128);
            config.num_envs = j["training"].value("num_envs", 10);
        }
        if (j.contains("logging")) {
            config.test_interval = j["logging"].value("test_interval", 10);
            config.test_trials = j["logging"].value("test_trials", 10);
            config.save_interval = j["logging"].value("save_interval", 200);
        }

        // Set Dimensions
        config.setEnvDimensions();
        return config;
    }

private:
    // Sets state_dim, action_dim, max_episodes, steps_per_ep based on env
    void setEnvDimensions(){
        if (env == "cartpole") {
            state_dim = 4;
            action_dim = 2;
            action_type = "discrete";
            max_episodes = 1000;
            steps_per_ep = 500;
        }
        else if (env == "lunarlander") {
            state_dim = 8;
            action_dim = 4;
            action_type = "discrete";
            max_episodes = 2000;
            steps_per_ep = 1000;
        }
        else if (env == "mountaincar") {
            state_dim = 2;
            action_dim = 1;
            action_type = "continuous";
            max_episodes = 2000;
            steps_per_ep = 999;
        }
        else {
            throw std::invalid_argument(
                "Unknown environment: " + env
            );
        }
    }
};

