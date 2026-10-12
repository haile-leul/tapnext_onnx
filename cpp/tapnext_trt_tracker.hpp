// TAPNext step model via a TensorRT engine (TensorRT 10 API, enqueueV3).
// Build the engine with tools/build_trt_engine.py (or trtexec, see README).
// See tapnext_points.hpp for the point/slot API.
//
//   TapNextTrtTracker tap("tapnext_step_256_fp16.engine", /*num_slots=*/64, roi);
//   auto& pts = tap.step(frame);
//
// The recurrent state (~150 MB for Q=64) lives on the GPU and is ping-ponged
// between two buffers; per frame only the 256x256 frame, the queries and the
// Q outputs cross PCIe. num_slots must be inside the engine's profile range.
#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>

#include "tapnext_points.hpp"

class TapNextTrtTracker : public TapNextPoints {
 public:
  TapNextTrtTracker(const std::string& engine_path, int num_slots, cv::Rect roi,
                    cudaStream_t stream = nullptr)
      : TapNextPoints(num_slots, roi), stream_(stream) {
    std::ifstream f(engine_path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + engine_path);
    std::string blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) throw std::runtime_error("engine deserialization failed (other GPU/TRT version?)");
    ctx_.reset(engine_->createExecutionContext());
    if (!stream_) {
      check(cudaStreamCreate(&stream_));
      own_stream_ = true;
    }

    const int n = kPatches + q_;
    using nvinfer1::Dims;
    auto dims = [](std::initializer_list<int64_t> d) {
      Dims r{};
      r.nbDims = int(d.size());
      int i = 0;
      for (auto v : d) r.d[i++] = v;
      return r;
    };
    setShape("frame", dims({1, kImg, kImg, 3}));
    setShape("query_points", dims({1, q_, 3}));
    setShape("step", dims({1}));
    setShape("video_reset", dims({1}));
    setShape("point_reset", dims({1, q_}));
    setShape("lru_state", dims({kLayers, n, kWidth}));
    setShape("conv_state", dims({kLayers, n, 3, kWidth}));

    d_frame_ = alloc(frame_.size());
    d_queries_ = alloc(queries_.size());
    d_step_ = alloc(1);
    d_vreset_ = alloc(1);
    d_preset_ = alloc(point_reset_.size());
    d_tracks_ = alloc(tracks_.size());
    d_vis_ = alloc(vis_.size());
    d_cert_ = alloc(cert_.size());
    for (int k = 0; k < 2; ++k) {
      d_lru_[k] = alloc(lruStateSize());
      d_conv_[k] = alloc(convStateSize());
    }
    resetState();
  }

  ~TapNextTrtTracker() override {
    for (void* p : allocs_) cudaFree(p);
    if (own_stream_) cudaStreamDestroy(stream_);
  }

  // Zero the recurrent state (call when restarting on a new video).
  void resetState() {
    for (int k = 0; k < 2; ++k) {
      check(cudaMemsetAsync(d_lru_[k], 0, lruStateSize() * sizeof(float), stream_));
      check(cudaMemsetAsync(d_conv_[k], 0, convStateSize() * sizeof(float), stream_));
    }
    check(cudaStreamSynchronize(stream_));
  }

 protected:
  void runModel(float stepf, float vreset) override {
    const int cur = ping_, nxt = 1 - ping_;
    h2d(d_frame_, frame_.data(), frame_.size());
    h2d(d_queries_, queries_.data(), queries_.size());
    h2d(d_step_, &stepf, 1);
    h2d(d_vreset_, &vreset, 1);
    h2d(d_preset_, point_reset_.data(), point_reset_.size());
    bind("frame", d_frame_);
    bind("query_points", d_queries_);
    bind("step", d_step_);
    bind("video_reset", d_vreset_);
    bind("point_reset", d_preset_);
    bind("lru_state", d_lru_[cur]);
    bind("conv_state", d_conv_[cur]);
    bind("tracks_xy", d_tracks_);
    bind("visible_prob", d_vis_);
    bind("certainty", d_cert_);
    bind("lru_state_out", d_lru_[nxt]);
    bind("conv_state_out", d_conv_[nxt]);
    if (!ctx_->enqueueV3(stream_)) throw std::runtime_error("enqueueV3 failed");
    d2h(tracks_.data(), d_tracks_, tracks_.size());
    d2h(vis_.data(), d_vis_, vis_.size());
    d2h(cert_.data(), d_cert_, cert_.size());
    check(cudaStreamSynchronize(stream_));
    ping_ = nxt;
  }

 private:
  struct Logger : nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
      if (s <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
    }
  };
  struct Del {
    template <class T>
    void operator()(T* p) const { delete p; }
  };

  static void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
  }
  float* alloc(size_t count) {
    void* p = nullptr;
    check(cudaMalloc(&p, count * sizeof(float)));
    allocs_.push_back(p);
    return static_cast<float*>(p);
  }
  void setShape(const char* name, const nvinfer1::Dims& d) {
    if (!ctx_->setInputShape(name, d))
      throw std::runtime_error(std::string("shape not allowed by engine profile: ") + name +
                               " (rebuild the engine with a Q range that includes num_slots)");
  }
  void bind(const char* name, void* ptr) {
    if (!ctx_->setTensorAddress(name, ptr))
      throw std::runtime_error(std::string("setTensorAddress failed: ") + name);
  }
  void h2d(float* dst, const float* src, size_t count) {
    check(cudaMemcpyAsync(dst, src, count * sizeof(float), cudaMemcpyHostToDevice, stream_));
  }
  void d2h(float* dst, const float* src, size_t count) {
    check(cudaMemcpyAsync(dst, src, count * sizeof(float), cudaMemcpyDeviceToHost, stream_));
  }

  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime, Del> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, Del> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, Del> ctx_;
  cudaStream_t stream_ = nullptr;
  bool own_stream_ = false;
  int ping_ = 0;
  std::vector<void*> allocs_;
  float *d_frame_, *d_queries_, *d_step_, *d_vreset_, *d_preset_, *d_tracks_, *d_vis_, *d_cert_;
  float* d_lru_[2];
  float* d_conv_[2];
};
