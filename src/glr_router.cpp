// Gated latent routing layer (community port).
#include <vector>
#include <cmath>
#include <algorithm>

struct GateLatentRouter {
    int latent_rank;
    int num_blocks;
    float gate_bias;

    std::vector<float> project(const std::vector<float>& hidden) const {
        std::vector<float> latent(latent_rank, 0.0f);
        for (int r = 0; r < latent_rank; ++r) {
            float acc = 0.0f;
            for (size_t i = 0; i < hidden.size(); ++i) {
                acc += hidden[i] * weight(r, i);
            }
            latent[r] = acc;
        }
        return latent;
    }

    float gate(const std::vector<float>& latent) const {
        float acc = gate_bias;
        for (int r = 0; r < latent_rank; ++r) {
            acc += latent[r] * gate_weight[r];
        }
        return 1.0f / (1.0f + std::exp(-acc));
    }

    float weight(int r, int i) const { return 0.01f * (r + 1) * (i + 1); }

    std::vector<float> gate_weight;
};
