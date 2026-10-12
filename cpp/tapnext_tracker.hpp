// TAPNext step model via ONNX Runtime (CPU or CUDA EP).
// See tapnext_points.hpp for the point/slot API.
//
//   TapNextTracker tap("tapnext_step_256_w16.onnx", /*num_slots=*/64, roi, /*use_cuda=*/true);
//
// Note: this version keeps the recurrent state in host memory (simple and
// portable). For GPU deployment prefer TapNextTrtTracker (tapnext_trt_tracker.hpp),
// which keeps the state on the device.
#pragma once

#include <onnxruntime_cxx_api.h>

#include <array>
#include <memory>
#include <string>

#include "tapnext_points.hpp"

class TapNextTracker : public TapNextPoints {
 public:
  TapNextTracker(const std::string& model_path, int num_slots, cv::Rect roi,
                 bool use_cuda = false)
      : TapNextPoints(num_slots, roi),
        env_(ORT_LOGGING_LEVEL_WARNING, "tapnext"),
        mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (use_cuda) {
      OrtCUDAProviderOptions cuda{};
      so.AppendExecutionProvider_CUDA(cuda);
    }
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), so);
    for (int k = 0; k < 2; ++k) {
      lru_[k].assign(lruStateSize(), 0.f);
      conv_[k].assign(convStateSize(), 0.f);
    }
  }

 protected:
  void runModel(float stepf, float vreset) override {
    const int cur = ping_, nxt = 1 - ping_;
    const int64_t n = kPatches + q_;
    std::array<int64_t, 4> fshape{1, kImg, kImg, 3};
    std::array<int64_t, 3> qshape{1, q_, 3};
    std::array<int64_t, 1> sshape{1};
    std::array<int64_t, 2> rshape{1, q_};
    std::array<int64_t, 3> lshape{kLayers, n, kWidth};
    std::array<int64_t, 4> cshape{kLayers, n, 3, kWidth};
    std::array<int64_t, 3> tshape{1, q_, 2};

    Ort::IoBinding io(*session_);
    io.BindInput("frame", mk(frame_.data(), frame_.size(), fshape));
    io.BindInput("query_points", mk(queries_.data(), queries_.size(), qshape));
    io.BindInput("step", mk(&stepf, 1, sshape));
    io.BindInput("video_reset", mk(&vreset, 1, sshape));
    io.BindInput("point_reset", mk(point_reset_.data(), point_reset_.size(), rshape));
    io.BindInput("lru_state", mk(lru_[cur].data(), lru_[cur].size(), lshape));
    io.BindInput("conv_state", mk(conv_[cur].data(), conv_[cur].size(), cshape));
    io.BindOutput("tracks_xy", mk(tracks_.data(), tracks_.size(), tshape));
    io.BindOutput("visible_prob", mk(vis_.data(), vis_.size(), rshape));
    io.BindOutput("certainty", mk(cert_.data(), cert_.size(), rshape));
    io.BindOutput("lru_state_out", mk(lru_[nxt].data(), lru_[nxt].size(), lshape));
    io.BindOutput("conv_state_out", mk(conv_[nxt].data(), conv_[nxt].size(), cshape));
    session_->Run(Ort::RunOptions{nullptr}, io);
    ping_ = nxt;
  }

 private:
  template <size_t R>
  Ort::Value mk(float* data, size_t count, const std::array<int64_t, R>& shape) {
    return Ort::Value::CreateTensor<float>(mem_, data, count, shape.data(), R);
  }

  Ort::Env env_;
  Ort::MemoryInfo mem_;
  std::unique_ptr<Ort::Session> session_;
  int ping_ = 0;
  std::array<std::vector<float>, 2> lru_, conv_;
};
