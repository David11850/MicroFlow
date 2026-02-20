#include "microflow/runtime.hpp"
#include <iostream>
#include <cstring>
#include <chrono>
#include <algorithm>

namespace microflow {

//==========================================================================
// 具体层实现
//==========================================================================

InputLayer::InputLayer(const std::string& name,
                      const std::vector<uint32_t>& shape)
    : name_(name), shape_(shape)
{
}

void InputLayer::forward(const std::vector<Tensor>& inputs,
                        std::vector<Tensor>& outputs,
                        float* workspace)
{
    // 输入层不做任何处理
    if (!inputs.empty() && !outputs.empty()) {
        inputs[0].copy_to(outputs[0].raw_ptr());
    }
}

std::vector<uint32_t> InputLayer::output_shape(
    const std::vector<std::vector<uint32_t>>& input_shapes
) const {
    return shape_;
}

//==========================================================================

Conv2DLayer::Conv2DLayer(const std::string& name,
                        const Tensor& kernel,
                        const Tensor& bias,
                        const Conv2DParams& params)
    : name_(name)
    , kernel_(kernel)
    , bias_(bias)
    , params_(params)
    , workspace_size_(0)
{
    workspace_size_ = compute_conv_workspace_size(
        Tensor({1, 100, 100}), kernel_, params_);
}

void Conv2DLayer::forward(const std::vector<Tensor>& inputs,
                         std::vector<Tensor>& outputs,
                         float* workspace)
{
    if (inputs.empty() || outputs.empty()) return;

    conv2d(inputs[0], kernel_, bias_, outputs[0], params_, workspace);
}

size_t Conv2DLayer::workspace_size() const {
    return workspace_size_;
}

std::vector<uint32_t> Conv2DLayer::output_shape(
    const std::vector<std::vector<uint32_t>>& input_shapes
) const {
    if (input_shapes.empty() || input_shapes[0].size() != 3) {
        return {};
    }

    int H = input_shapes[0][1];
    int W = input_shapes[0][2];
    int K = params_.kernel_size;
    int F = kernel_.shapes()[0];

    int H_out = (H + 2 * params_.padding - K) / params_.stride + 1;
    int W_out = (W + 2 * params_.padding - K) / params_.stride + 1;

    return {static_cast<uint32_t>(F), static_cast<uint32_t>(H_out),
            static_cast<uint32_t>(W_out)};
}

//==========================================================================

DepthwiseConv2DLayer::DepthwiseConv2DLayer(const std::string& name,
                                          const Tensor& kernel,
                                          const Tensor& bias,
                                          const Conv2DParams& params)
    : name_(name), kernel_(kernel), bias_(bias), params_(params)
{
}

void DepthwiseConv2DLayer::forward(const std::vector<Tensor>& inputs,
                                  std::vector<Tensor>& outputs,
                                  float* workspace)
{
    if (inputs.empty() || outputs.empty()) return;

    // 使用depthwise卷积
    conv2d(inputs[0], kernel_, bias_, outputs[0], params_);
}

std::vector<uint32_t> DepthwiseConv2DLayer::output_shape(
    const std::vector<std::vector<uint32_t>>& input_shapes
) const {
    if (input_shapes.empty() || input_shapes[0].size() != 3) {
        return {};
    }

    int H = input_shapes[0][1];
    int W = input_shapes[0][2];
    int K = params_.kernel_size;
    int C = input_shapes[0][0];

    int H_out = (H + 2 * params_.padding - K) / params_.stride + 1;
    int W_out = (W + 2 * params_.padding - K) / params_.stride + 1;

    return {static_cast<uint32_t>(C), static_cast<uint32_t>(H_out),
            static_cast<uint32_t>(W_out)};
}

//==========================================================================

BatchNormLayer::BatchNormLayer(const std::string& name,
                              const Tensor& mean,
                              const Tensor& var,
                              const Tensor& gamma,
                              const Tensor& beta,
                              float eps)
    : name_(name), mean_(mean), var_(var), gamma_(gamma), beta_(beta), eps_(eps)
{
}

void BatchNormLayer::forward(const std::vector<Tensor>& inputs,
                            std::vector<Tensor>& outputs,
                            float* workspace)
{
    if (inputs.empty() || outputs.empty()) return;

    // 复制输入到输出
    inputs[0].copy_to(outputs[0].raw_ptr());

    // 应用BatchNorm
    batch_norm(outputs[0], mean_, var_, gamma_, beta_, eps_);
}

std::vector<uint32_t> BatchNormLayer::output_shape(
    const std::vector<std::vector<uint32_t>>& input_shapes
) const {
    return input_shapes.empty() ? std::vector<uint32_t>() : input_shapes[0];
}

//==========================================================================

ActivationLayer::ActivationLayer(const std::string& name, LayerType type)
    : name_(name), activation_type_(type)
{
}

void ActivationLayer::forward(const std::vector<Tensor>& inputs,
                             std::vector<Tensor>& outputs,
                             float* workspace)
{
    if (inputs.empty() || outputs.empty()) return;

    // 复制输入到输出 (如果不是就地操作)
    if (inputs[0].raw_ptr() != outputs[0].raw_ptr()) {
        inputs[0].copy_to(outputs[0].raw_ptr());
    }

    // 应用激活函数
    switch (activation_type_) {
        case LayerType::kReLU:
            relu(outputs[0]);
            break;
        case LayerType::kReLU6:
            relu6(outputs[0]);
            break;
        case LayerType::kLeakyReLU:
            leaky_relu(outputs[0], 0.01f);
            break;
        case LayerType::kELU:
            elu(outputs[0]);
            break;
        case LayerType::kSigmoid:
            sigmoid(outputs[0]);
            break;
        default:
            break;
    }
}

std::vector<uint32_t> ActivationLayer::output_shape(
    const std::vector<std::vector<uint32_t>>& input_shapes
) const {
    return input_shapes.empty() ? std::vector<uint32_t>() : input_shapes[0];
}

//==========================================================================

PoolingLayer::PoolingLayer(const std::string& name, LayerType type,
                          int kernel_size, int stride, int padding)
    : name_(name)
    , pool_type_(type)
    , kernel_size_(kernel_size)
    , stride_(stride == 0 ? kernel_size : stride)
    , padding_(padding)
{
}

void PoolingLayer::forward(const std::vector<Tensor>& inputs,
                          std::vector<Tensor>& outputs,
                          float* workspace)
{
    if (inputs.empty() || outputs.empty()) return;

    switch (pool_type_) {
        case LayerType::kMaxPool2D:
            max_pool2d(inputs[0], outputs[0], kernel_size_, stride_, padding_);
            break;
        case LayerType::kAvgPool2D:
            avg_pool2d(inputs[0], outputs[0], kernel_size_, stride_, padding_);
            break;
        case LayerType::kGlobalAvgPool2D:
            global_avg_pool2d(inputs[0], outputs[0]);
            break;
        case LayerType::kAdaptiveAvgPool2D:
            adaptive_avg_pool2d(inputs[0], outputs[0],
                               outputs[0].shapes()[1],
                               outputs[0].shapes()[2]);
            break;
        default:
            break;
    }
}

std::vector<uint32_t> PoolingLayer::output_shape(
    const std::vector<std::vector<uint32_t>>& input_shapes
) const {
    if (input_shapes.empty() || input_shapes[0].size() != 3) {
        return {};
    }

    int C = input_shapes[0][0];
    int H = input_shapes[0][1];
    int W = input_shapes[0][2];

    if (pool_type_ == LayerType::kGlobalAvgPool2D) {
        return {static_cast<uint32_t>(C), 1, 1};
    }

    int H_out = (H + 2 * padding_ - kernel_size_) / stride_ + 1;
    int W_out = (W + 2 * padding_ - kernel_size_) / stride_ + 1;

    return {static_cast<uint32_t>(C), static_cast<uint32_t>(H_out),
            static_cast<uint32_t>(W_out)};
}

//==========================================================================

LinearLayer::LinearLayer(const std::string& name,
                        const Tensor& weight,
                        const Tensor& bias)
    : name_(name), weight_(weight), bias_(bias)
{
}

void LinearLayer::forward(const std::vector<Tensor>& inputs,
                         std::vector<Tensor>& outputs,
                         float* workspace)
{
    if (inputs.empty() || outputs.empty()) return;

    linear(inputs[0], weight_, bias_, outputs[0]);
}

std::vector<uint32_t> LinearLayer::output_shape(
    const std::vector<std::vector<uint32_t>>& input_shapes
) const {
    if (input_shapes.empty()) {
        return {weight_.shapes()[0]};
    }

    // 计算展平后的输入大小
    uint32_t input_size = 1;
    for (auto dim : input_shapes[0]) {
        input_size *= dim;
    }

    return {input_size / weight_.shapes()[1], weight_.shapes()[0]};
}

//==========================================================================

ReshapeLayer::ReshapeLayer(const std::string& name,
                          const std::vector<uint32_t>& shape)
    : name_(name), shape_(shape)
{
}

void ReshapeLayer::forward(const std::vector<Tensor>& inputs,
                          std::vector<Tensor>& outputs,
                          float* workspace)
{
    if (inputs.empty() || outputs.empty()) return;

    // Reshape是零拷贝操作
    outputs[0] = inputs[0].reshape(shape_);
}

std::vector<uint32_t> ReshapeLayer::output_shape(
    const std::vector<std::vector<uint32_t>>& input_shapes
) const {
    return shape_;
}

//==========================================================================

FlattenLayer::FlattenLayer(const std::string& name)
    : name_(name)
{
}

void FlattenLayer::forward(const std::vector<Tensor>& inputs,
                          std::vector<Tensor>& outputs,
                          float* workspace)
{
    if (inputs.empty() || outputs.empty()) return;

    flatten(inputs[0], outputs[0]);
}

std::vector<uint32_t> FlattenLayer::output_shape(
    const std::vector<std::vector<uint32_t>>& input_shapes
) const {
    if (input_shapes.empty()) {
        return {};
    }

    uint32_t size = 1;
    for (auto dim : input_shapes[0]) {
        size *= dim;
    }

    return {size};
}

//==========================================================================

SoftmaxLayer::SoftmaxLayer(const std::string& name, int axis)
    : name_(name), axis_(axis)
{
}

void SoftmaxLayer::forward(const std::vector<Tensor>& inputs,
                          std::vector<Tensor>& outputs,
                          float* workspace)
{
    if (inputs.empty() || outputs.empty()) return;

    // 复制输入
    if (inputs[0].raw_ptr() != outputs[0].raw_ptr()) {
        inputs[0].copy_to(outputs[0].raw_ptr());
    }

    // 应用softmax
    softmax(outputs[0], axis_);
}

std::vector<uint32_t> SoftmaxLayer::output_shape(
    const std::vector<std::vector<uint32_t>>& input_shapes
) const {
    return input_shapes.empty() ? std::vector<uint32_t>() : input_shapes[0];
}

//==========================================================================
// 模型实现
//==========================================================================

Model::Model()
    : is_loaded_(false)
{
}

Model::~Model() = default;

bool Model::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open model file: " << path << "\n";
        return false;
    }

    // 读取文件头
    ModelHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(ModelHeader));

    // 验证魔数
    if (header.magic != MFLOW_MAGIC) {
        std::cerr << "Error: Invalid model file format\n";
        return false;
    }

    std::cout << "Loading model: " << header.description << "\n";
    std::cout << "Version: " << header.version << "\n";
    std::cout << "Layers: " << header.num_layers << "\n";

    // 简化实现: 这里应该解析完整的模型文件
    // 实际实现需要读取层定义、权重等

    is_loaded_ = true;
    return true;
}

bool Model::save(const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // 写入文件头
    ModelHeader header;
    header.magic = MFLOW_MAGIC;
    header.version = 2;
    header.num_layers = layers_.size();
    header.num_tensors = 0;
    std::strncpy(header.description, name_.c_str(), 63);

    file.write(reinterpret_cast<const char*>(&header), sizeof(ModelHeader));

    // 写入层数据...
    // 简化实现

    return true;
}

void Model::add_layer(std::unique_ptr<Layer> layer) {
    layer_map_[layer->name()] = layer.get();
    layers_.push_back(std::move(layer));
}

Layer* Model::get_layer(const std::string& name) {
    auto it = layer_map_.find(name);
    return it != layer_map_.end() ? it->second : nullptr;
}

void Model::forward(const Tensor& input, Tensor& output) {
    if (!is_loaded_ || layers_.empty()) {
        std::cerr << "Error: Model not loaded or empty\n";
        return;
    }

    // 分配中间张量
    if (intermediate_tensors_.empty()) {
        allocate_tensors();
    }

    // 计算工作空间
    if (workspace_.empty()) {
        size_t ws_size = compute_workspace_size();
        workspace_.resize(ws_size / sizeof(float));
    }

    // 第一层的输入是用户输入
    std::vector<Tensor> current_input = {input};
    std::vector<Tensor> current_output;

    // 逐层执行
    for (size_t i = 0; i < layers_.size(); ++i) {
        Layer* layer = layers_[i].get();

        // 获取输出张量
        if (i < intermediate_tensors_.size()) {
            current_output = {intermediate_tensors_[i]};
        }

        // 执行前向传播
        layer->forward(current_input, current_output, workspace_.data());

        // 下一层的输入是当前层的输出
        current_input = current_output;
    }

    // 最后一个输出是结果
    if (!current_output.empty()) {
        current_output[0].copy_to(output.raw_ptr());
    }
}

void Model::forward_batch(const std::vector<Tensor>& inputs,
                         std::vector<Tensor>& outputs)
{
    for (size_t i = 0; i < inputs.size(); ++i) {
        forward(inputs[i], outputs[i]);
    }
}

std::vector<uint32_t> Model::input_shape() const {
    return input_shape_;
}

std::vector<uint32_t> Model::output_shape() const {
    return output_shape_;
}

auto Model::get_info() const -> Info {
    Info info;
    info.name = name_;
    info.description = "";
    info.num_layers = layers_.size();
    info.num_parameters = 0;
    info.model_size = 0;
    return info;
}

void Model::summary() const {
    std::cout << "========================================\n";
    std::cout << "Model Summary: " << name_ << "\n";
    std::cout << "========================================\n";
    std::cout << "Layers: " << layers_.size() << "\n";

    for (const auto& layer : layers_) {
        std::cout << "  - " << layer->name() << "\n";
    }

    std::cout << "========================================\n";
}

void Model::allocate_tensors() {
    // 简化实现
    intermediate_tensors_.clear();
    // 实际实现需要根据每层的输出形状分配
}

size_t Model::compute_workspace_size() {
    size_t max_size = 0;
    for (const auto& layer : layers_) {
        max_size = std::max(max_size, layer->workspace_size());
    }
    return max_size;
}

void Model::fuse_layers() {
    // 简化实现
    // 实际实现应该检测Conv+BN+ReLU等可融合的模式
}

//==========================================================================
// 模型构建器实现
//==========================================================================

ModelBuilder::ModelBuilder(const std::string& name)
    : name_(name)
{
}

ModelBuilder& ModelBuilder::input(const std::vector<uint32_t>& shape) {
    current_shape_ = shape;
    return *this;
}

ModelBuilder& ModelBuilder::conv2d(const std::string& name,
                                  int out_channels,
                                  int kernel_size,
                                  int stride,
                                  int padding,
                                  bool bias)
{
    // 简化实现
    return *this;
}

ModelBuilder& ModelBuilder::depthwise_conv2d(const std::string& name,
                                           int kernel_size,
                                           int stride,
                                           int padding,
                                           bool bias)
{
    return *this;
}

ModelBuilder& ModelBuilder::batch_norm(const std::string& name) {
    return *this;
}

ModelBuilder& ModelBuilder::relu() {
    return *this;
}

ModelBuilder& ModelBuilder::relu6() {
    return *this;
}

ModelBuilder& ModelBuilder::leaky_relu(float alpha) {
    return *this;
}

ModelBuilder& ModelBuilder::max_pool(int kernel_size, int stride, int padding) {
    return *this;
}

ModelBuilder& ModelBuilder::avg_pool(int kernel_size, int stride, int padding) {
    return *this;
}

ModelBuilder& ModelBuilder::global_avg_pool() {
    return *this;
}

ModelBuilder& ModelBuilder::linear(const std::string& name,
                                  int out_features,
                                  bool bias)
{
    return *this;
}

ModelBuilder& ModelBuilder::flatten() {
    return *this;
}

ModelBuilder& ModelBuilder::reshape(const std::vector<uint32_t>& shape) {
    return *this;
}

ModelBuilder& ModelBuilder::softmax(int axis) {
    return *this;
}

Model ModelBuilder::build() {
    Model model;
    model.name_ = name_;

    // 转移所有权
    for (auto& layer : layers_) {
        model.add_layer(std::move(layer));
    }

    return model;
}

//==========================================================================
// 推理引擎实现
//==========================================================================

InferenceEngine::InferenceEngine(const Config& config)
    : config_(config)
{
    // 初始化统计
    std::memset(&stats_, 0, sizeof(stats_));
}

bool InferenceEngine::load_model(const std::string& path) {
    return model_.load(path);
}

Tensor InferenceEngine::infer(const Tensor& input) {
    auto start = std::chrono::high_resolution_clock::now();

    // 创建输出张量
    Tensor output = Tensor::zeros(model_.output_shape());

    // 执行推理
    model_.forward(input, output);

    // 记录时间
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    inference_times_.push_back(time_ms);
    stats_.num_inferences++;

    return output;
}

std::vector<Tensor> InferenceEngine::infer_batch(
    const std::vector<Tensor>& inputs)
{
    std::vector<Tensor> outputs(inputs.size());

    #pragma omp parallel for
    for (size_t i = 0; i < inputs.size(); ++i) {
        outputs[i] = infer(inputs[i]);
    }

    return outputs;
}

auto InferenceEngine::get_stats() const -> Stats {
    Stats stats = stats_;

    if (!inference_times_.empty()) {
        stats.total_time_ms = 0;
        stats.min_time_ms = inference_times_[0];
        stats.max_time_ms = inference_times_[0];

        for (double t : inference_times_) {
            stats.total_time_ms += t;
            stats.min_time_ms = std::min(stats.min_time_ms, t);
            stats.max_time_ms = std::max(stats.max_time_ms, t);
        }

        stats.avg_time_ms = stats.total_time_ms / inference_times_.size();
        stats.throughput = 1000.0 / stats.avg_time_ms;
    }

    return stats;
}

void InferenceEngine::reset_stats() {
    std::memset(&stats_, 0, sizeof(stats_));
    inference_times_.clear();
}

//==========================================================================
// 辅助函数
//==========================================================================

Model create_mnist_model() {
    Model model;
    model.name_ = "MNIST_Classifier";

    // 简化实现
    // 实际应该构建完整的LeNet或类似模型

    return model;
}

Model create_mobilenet_v2_model(int alpha) {
    Model model;
    model.name_ = "MobileNetV2";
    return model;
}

Model create_simple_cnn_model(const std::vector<uint32_t>& input_shape,
                             int num_classes)
{
    Model model;
    model.name_ = "SimpleCNN";
    return model;
}

} // namespace microflow
