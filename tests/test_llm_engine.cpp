#include "llm_engine.h"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace microflow;

int main() {
    Pi4Profile profile;
    profile.hidden_size = 8;
    profile.vocab_size = 16;
    profile.max_seq_len = 32;
    profile.num_threads = 2;

    Pi4LlmEngine engine(profile);

    std::vector<float> embedding(static_cast<size_t>(profile.vocab_size) * profile.hidden_size, 0.0f);
    for (uint32_t t = 0; t < profile.vocab_size; ++t) {
        for (uint32_t h = 0; h < profile.hidden_size; ++h) {
            embedding[static_cast<size_t>(t) * profile.hidden_size + h] = static_cast<float>((t + h) % 7) / 7.0f;
        }
    }
    engine.load_embedding_table(embedding);

    std::vector<float> lm_head_fp(static_cast<size_t>(profile.vocab_size) * profile.hidden_size, 0.0f);
    for (uint32_t r = 0; r < profile.vocab_size; ++r) {
        for (uint32_t c = 0; c < profile.hidden_size; ++c) {
            lm_head_fp[static_cast<size_t>(r) * profile.hidden_size + c] = (r == 3 ? 0.8f : 0.2f) + 0.01f * c;
        }
    }
    auto q_lm_head = QuantizedMatrix::from_float(lm_head_fp, profile.vocab_size, profile.hidden_size);
    engine.load_lm_head(q_lm_head);

    std::vector<int> prompt = {1, 2};
    auto out = engine.generate(prompt, 4);

    assert(out.size() >= prompt.size());
    for (int tk : out) {
        assert(tk >= 0 && tk < static_cast<int>(profile.vocab_size));
    }

    KvCache cache(4, 4);
    cache.append({1, 0, 0, 0}, {1, 0, 0, 0});
    cache.append({0, 1, 0, 0}, {0, 1, 0, 0});
    auto attn = cache.attention({1, 0, 0, 0});
    assert(attn[0] > attn[1]);

    std::cout << "test_llm_engine passed, generated tokens: ";
    for (int tk : out) std::cout << tk << " ";
    std::cout << std::endl;
    return 0;
}
