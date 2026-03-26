#include "CartPoleEnv.h"
#include <cmath>
#include <stdexcept>

// Constructor
CartPoleEnv::CartPoleEnv(int seed):

    cartPosition(0.0f),
    cartVelocity(0.0f),
    poleAngle(0.0f),
    poleAngularVelocity(0.0f),
    stepCount(0),
    done(false),

    rng(seed == -1 ? std::random_device{}() : static_cast<unsigned int>(seed)),
    initDist(-0.05f, 0.05f)
{}

// reset()
std::vector<float> CartPoleEnv::reset() {
    cartPosition        = initDist(rng);
    cartVelocity        = initDist(rng);
    poleAngle           = initDist(rng);
    poleAngularVelocity = initDist(rng);

    stepCount = 0;
    done = false;

    return getObservation();
}

// step()
StepResult CartPoleEnv::step(int action) {
    if (action != 0 && action != 1) {
        throw std::invalid_argument("Action must be 0 or 1");
    }
    if (done) {
        throw std::runtime_error("Cannot step in a finished episode. Call reset() first.");
    }

    // Physics
    float force = (action == 1) ? FORCE_MAG : -FORCE_MAG;

    float cosTheta = std::cos(poleAngle);
    float sinTheta = std::sin(poleAngle);

    float totalMass = CART_MASS + POLE_MASS;
    float poleMassLen = POLE_MASS * POLE_HALF_LEN;

    // intermediate calc
    float temp = (force + poleMassLen * poleAngularVelocity * poleAngularVelocity * sinTheta) / totalMass;

    // Angular acceleration of the pole
    float poleAngularAccel = (GRAVITY * sinTheta - cosTheta * temp)
    / (POLE_HALF_LEN * (4.0f / 3.0f - POLE_MASS * cosTheta * cosTheta / totalMass));

    float cartAccel = temp - poleMassLen * poleAngularAccel * cosTheta / totalMass;

    // Euler Integration
    cartVelocity += cartAccel * DT;
    cartPosition += cartVelocity * DT;
    poleAngularVelocity += poleAngularAccel * DT;
    poleAngle += poleAngularVelocity * DT;

    // Update Episode State
    stepCount++;
    done = checkDone();

    // Build & Return result
    StepResult result;
    result.observation = getObservation();
    result.reward = 1.0f;
    result.done = done;
    return result;
}

// getObservation()
std::vector<float> CartPoleEnv::getObservation() const {
    return {cartPosition, cartVelocity, poleAngle, poleAngularVelocity};
}

//getStepCount()
int CartPoleEnv::getStepCount() const {
    return stepCount;
}

// isDone()
bool CartPoleEnv::isDone() const {
    return done;
}

// checkDone()
bool CartPoleEnv::checkDone() const {
    // episode end if the pole has fallen too far
    if (std::fabs(poleAngle) > MAX_ANGLE) return true;

    // Episode ends if cart goes off track
    if (std::fabs(cartPosition) > MAX_POSITION) return true;

    // episode ends if max steps reached
    if (stepCount >= MAX_STEPS) return true;

    return false;
}

