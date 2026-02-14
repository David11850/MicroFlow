#include "llm_engine.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace microflow {

QuantizedMatrix QuantizedMatrix::from_float(const std::vector<float>& src, uint32_t rows, uint32_t cols) {
    if (src.size() != static_cast<size_t>(rows) * cols) {
        throw std::runtime_error("QuantizedMatrix::from_float shape mismatch");
    }

    QuantizedMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.weight.resize(src.size());
    q.scales.resize(rows, 1.0f);

    for (uint32_t r = 0; r < rows; ++r) {
        const float* row_ptr = src.data() + static_cast<size_t>(r) * cols;
        float max_abs = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            max_abs = std::max(max_abs, std::abs(row_ptr[c]));
        }
        float scale = (max_abs < 1e-8f) ? 1.0f : max_abs / 127.0f;
        q.scales[r] = scale;

        for (uint32_t c = 0; c < cols; ++c) {
            int quant = static_cast<int>(std::round(row_ptr[c] / scale));
            quant = std::max(-127, std::min(127, quant));
            q.weight[static_cast<size_t>(r) * cols + c] = static_cast<int8_t>(quant);
        }
    }

    return q;
}

KvCache::KvCache(uint32_t max_seq_len, uint32_t hidden_size)
    : m_max_seq_len(max_seq_len),
      m_hidden_size(hidden_size),
      m_seq_len(0),
      m_keys(static_cast<size_t>(max_seq_len) * hidden_size, 0.0f),
      m_values(static_cast<size_t>(max_seq_len) * hidden_size, 0.0f) {}

void KvCache::append(const std::vector<float>& key, const std::vector<float>& value) {
    if (key.size() != m_hidden_size || value.size() != m_hidden_size) {
        throw std::runtime_error("KvCache::append hidden size mismatch");
    }
    if (m_seq_len >= m_max_seq_len) {
        throw std::runtime_error("KvCache full");
    }

    float* key_dst = m_keys.data() + static_cast<size_t>(m_seq_len) * m_hidden_size;
    float* val_dst = m_values.data() + static_cast<size_t>(m_seq_len) * m_hidden_size;
    std::copy(key.begin(), key.end(), key_dst);
    std::copy(value.begin(), value.end(), val_dst);
    ++m_seq_len;
}

std::vector<float> KvCache::attention(const std::vector<float>& query) const {
    if (query.size() != m_hidden_size) {
        throw std::runtime_error("KvCache::attention hidden size mismatch");
    }
    if (m_seq_len == 0) {
        return std::vector<float>(m_hidden_size, 0.0f);
    }

    std::vector<float> scores(m_seq_len, 0.0f);
    const float inv_sqrt_dim = 1.0f / std::sqrt(static_cast<float>(m_hidden_size));

    for (uint32_t t = 0; t < m_seq_len; ++t) {
        const float* key_ptr = m_keys.data() + static_cast<size_t>(t) * m_hidden_size;
        float dot = 0.0f;
        for (uint32_t i = 0; i < m_hidden_size; ++i) {
            dot += query[i] * key_ptr[i];
        }
        scores[t] = dot * inv_sqrt_dim;
    }

    float max_score = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (float& s : scores) {
        s = std::exp(s - max_score);
        sum += s;
    }
    for (float& s : scores) {
        s /= sum;
    }

    std::vector<float> out(m_hidden_size, 0.0f);
    for (uint32_t t = 0; t < m_seq_len; ++t) {
        const float* val_ptr = m_values.data() + static_cast<size_t>(t) * m_hidden_size;
        float w = scores[t];
        for (uint32_t i = 0; i < m_hidden_size; ++i) {
            out[i] += w * val_ptr[i];
        }
    }
    return out;
}

Pi4LlmEngine::Pi4LlmEngine(Pi4Profile profile) : m_profile(profile) {
#ifdef _OPENMP
    omp_set_num_threads(static_cast<int>(m_profile.num_threads));
#endif
}

void Pi4LlmEngine::load_embedding_table(std::vector<float> embeddings) {
    const size_t expected = static_cast<size_t>(m_profile.vocab_size) * m_profile.hidden_size;
    if (embeddings.size() != expected) {
        throw std::runtime_error("Embedding table shape mismatch");
    }
    m_embeddings = std::move(embeddings);
}

void Pi4LlmEngine::load_lm_head(const QuantizedMatrix& lm_head) {
    if (lm_head.rows != m_profile.vocab_size || lm_head.cols != m_profile.hidden_size) {
        throw std::runtime_error("LM head shape mismatch");
    }
    m_lm_head = lm_head;
}

std::vector<float> Pi4LlmEngine::embed(uint32_t token_id) const {
    if (token_id >= m_profile.vocab_size) {
        throw std::runtime_error("Token id out of range");
    }
    const float* start = m_embeddings.data() + static_cast<size_t>(token_id) * m_profile.hidden_size;
    return std::vector<float>(start, start + m_profile.hidden_size);
}

std::vector<float> Pi4LlmEngine::quantized_matvec(const QuantizedMatrix& mat, const std::vector<float>& x) const {
    if (x.size() != mat.cols) {
        throw std::runtime_error("quantized_matvec input size mismatch");
    }

    std::vector<float> y(mat.rows, 0.0f);

#pragma omp parallel for if(mat.rows > 64)
    for (uint32_t r = 0; r < mat.rows; ++r) {
        const int8_t* w = mat.weight.data() + static_cast<size_t>(r) * mat.cols;
        float sum = 0.0f;

#if defined(__aarch64__) || defined(_M_ARM64)
        uint32_t c = 0;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (; c + 4 <= mat.cols; c += 4) {
            int8x8_t wi8 = vld1_s8(w + c);
            int16x8_t wi16 = vmovl_s8(wi8);
            int32x4_t wi32 = vmovl_s16(vget_low_s16(wi16));
            float32x4_t wf = vcvtq_f32_s32(wi32);
            float32x4_t xv = vld1q_f32(x.data() + c);
            acc = vmlaq_f32(acc, wf, xv);
        }
        float32x2_t pair = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        sum += vget_lane_f32(vpadd_f32(pair, pair), 0);
        for (; c < mat.cols; ++c) {
            sum += static_cast<float>(w[c]) * x[c];
        }
#else
        for (uint32_t c = 0; c < mat.cols; ++c) {
            sum += static_cast<float>(w[c]) * x[c];
        }
#endif

        y[r] = sum * mat.scales[r];
    }

    return y;
}

int Pi4LlmEngine::greedy_sample(const std::vector<float>& logits) {
    return static_cast<int>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

std::vector<int> Pi4LlmEngine::generate(const std::vector<int>& prompt_tokens, uint32_t max_new_tokens) const {
    if (m_embeddings.empty() || m_lm_head.weight.empty()) {
        throw std::runtime_error("Model not fully loaded");
    }

    std::vector<int> all_tokens = prompt_tokens;
    KvCache cache(m_profile.max_seq_len, m_profile.hidden_size);

    for (int token : prompt_tokens) {
        auto h = embed(static_cast<uint32_t>(token));
        cache.append(h, h);
    }

    for (uint32_t step = 0; step < max_new_tokens; ++step) {
        std::vector<float> query = embed(static_cast<uint32_t>(all_tokens.back()));
        std::vector<float> ctx = cache.attention(query);
        for (uint32_t i = 0; i < m_profile.hidden_size; ++i) {
            query[i] = 0.5f * query[i] + 0.5f * ctx[i];
        }

        std::vector<float> logits = quantized_matvec(m_lm_head, query);
        int next_token = greedy_sample(logits);

        all_tokens.push_back(next_token);
        auto kv = embed(static_cast<uint32_t>(next_token));
        cache.append(kv, kv);

        if (cache.size() >= m_profile.max_seq_len) {
            break;
        }
    }

    return all_tokens;
}

} // namespace microflow
