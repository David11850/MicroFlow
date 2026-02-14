#ifndef LLM_ENGINE_H
#define LLM_ENGINE_H

#include <cstdint>
#include <vector>

namespace microflow {

struct Pi4Profile {
    uint32_t num_threads = 4;
    uint32_t max_seq_len = 256;
    uint32_t hidden_size = 256;
    uint32_t vocab_size = 32000;
    float temperature = 0.7f;
};

struct QuantizedMatrix {
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::vector<int8_t> weight;
    std::vector<float> scales;

    static QuantizedMatrix from_float(const std::vector<float>& src, uint32_t rows, uint32_t cols);
};

class KvCache {
public:
    KvCache(uint32_t max_seq_len, uint32_t hidden_size);
    void append(const std::vector<float>& key, const std::vector<float>& value);
    std::vector<float> attention(const std::vector<float>& query) const;
    uint32_t size() const { return m_seq_len; }

private:
    uint32_t m_max_seq_len;
    uint32_t m_hidden_size;
    uint32_t m_seq_len;
    std::vector<float> m_keys;
    std::vector<float> m_values;
};

class Pi4LlmEngine {
public:
    explicit Pi4LlmEngine(Pi4Profile profile);

    void load_embedding_table(std::vector<float> embeddings);
    void load_lm_head(const QuantizedMatrix& lm_head);

    std::vector<int> generate(const std::vector<int>& prompt_tokens, uint32_t max_new_tokens) const;

private:
    std::vector<float> embed(uint32_t token_id) const;
    std::vector<float> quantized_matvec(const QuantizedMatrix& mat, const std::vector<float>& x) const;
    static int greedy_sample(const std::vector<float>& logits);

    Pi4Profile m_profile;
    std::vector<float> m_embeddings;
    QuantizedMatrix m_lm_head;
};

} // namespace microflow

#endif
