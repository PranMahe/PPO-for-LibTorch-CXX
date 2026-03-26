#pragma once

#include <vector>
#include <random>

struct StepResult {
    std::vector<float> observation;
    float reward;
    bool done;
};

class CartPoleEnv {
public:

    //Physical Constants
    static constexpr float GRAVITY          = 9.8f;
    static constexpr float CART_MASS        = 1.0f;
    static constexpr float POLE_MASS        = 0.1f;
    static constexpr float POLE_HALF_LEN    = 0.5f;
    static constexpr float FORCE_MAG        = 10.0f;
    static constexpr float DT               = 0.02f;

    // Episode termination thresholds
    static constexpr float MAX_ANGLE        = 12.0f * 3.1415926f / 180.0f;
    static constexpr float MAX_POSITION     = 2.4f;
    static constexpr int   MAX_STEPS        = 500;

    // Constructor
    explicit CartPoleEnv(int seed = -1);

    // Destructor
    ~CartPoleEnv() = default;

    // Core Interface
    std::vector<float> reset();

    StepResult step (int action);

    // Read-only Accessors
    std::vector<float> getObservation() const;
    int getStepCount() const;
    bool isDone() const;

private:

    // State Variables
    float cartPosition;
    float cartVelocity;
    float poleAngle;
    float poleAngularVelocity;

    // Episode Tracking
    int stepCount;
    bool done;

    // Random Num Generation
    std::mt19937 rng;
    std::uniform_real_distribution<float> initDist;

    // Private Helper
    bool checkDone() const;
};