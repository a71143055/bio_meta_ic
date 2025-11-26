#include "fast_update.h"

void fast_update(float* state, float* inputs, float* leak, float* bias, float dt, int n) {
    // Leaky integrator: s += dt * (-leak*s + inputs + bias)
    for (int i = 0; i < n; ++i) {
        float ds = (-leak[i] * state[i]) + inputs[i] + bias[i];
        state[i] += dt * ds;
        // Optional saturation to mimic biological bounds
        if (state[i] > 5.0f) state[i] = 5.0f;
        if (state[i] < -5.0f) state[i] = -5.0f;
    }
}
