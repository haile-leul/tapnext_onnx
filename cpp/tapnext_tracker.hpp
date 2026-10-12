// Minimal C++ wrapper around tapnext_step_256.onnx (ONNX Runtime + OpenCV).
//
// The ONNX graph runs one frame and carries the recurrent state explicitly.
// This class owns that state, a fixed pool of point "slots", and returns each
// active point's position and velocity in ORIGINAL image pixels.
//
// Typical use (vehicle-assisted face velocity init):
//   TapNextTracker tap("tapnext_step_256.onnx", /*num_slots=*/64, roi);
//   for each frame:
//     for each newly confirmed vehicle track: tap.addPoint(x, y) for a few
//         points inside the vehicle box (e.g. a 3x3 grid around the windshield)
//     auto pts = tap.step(frame);
//     -> pts[i].vx, pts[i].vy are px/frame; median over the points inside a
//        vehicle box gives the motion to seed a new face track's Kalman state.
#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <vector>

class TapNextTracker {
 public:
  static constexpr int kImg = 256;
  static constexpr int kLayers = 12;
  static constexpr int kPatches = 1024;
  static constexpr int kWidth = 768;
  static constexpr float kInactiveT = 1e9f;  // query time "in the future"

  struct Point {
    int slot = -1;
    bool active = false;
    float x = 0, y = 0;     // position, original image pixels
    float vx = 0, vy = 0;   // velocity, original image pixels / frame
    float visible = 0;      // sigmoid(visibility logit)
    float certainty = 0;    // probability mass within 8px (at 256 res)
    bool reliable() const { return visible * certainty > 0.5f; }
  };

  // roi: region of the original frame fed to the model (e.g. the gate lane).
  // It is resized to 256x256, so a tighter ROI = higher effective resolution.
  TapNextTracker(const std::string& model_path, int num_slots, cv::Rect roi,
                 bool use_cuda = false)
      : env_(ORT_LOGGING_LEVEL_WARNING, "tapnext"),
        q_(num_slots),
        roi_(roi),
        mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (use_cuda) {
      OrtCUDAProviderOptions cuda{};
      so.AppendExecutionProvider_CUDA(cuda);
    }
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), so);

    const size_t n = kPatches + q_;
    frame_.assign(kImg * kImg * 3, 0.f);
    queries_.assign(q_ * 3, 0.f);
    point_reset_.assign(q_, 0.f);
    for (int i = 0; i < q_; ++i) queries_[i * 3] = kInactiveT;
    for (int k = 0; k < 2; ++k) {
      lru_[k].assign(size_t(kLayers) * n * kWidth, 0.f);
      conv_[k].assign(size_t(kLayers) * n * 3 * kWidth, 0.f);
    }
    tracks_.assign(q_ * 2, 0.f);
    vis_.assign(q_, 0.f);
    cert_.assign(q_, 0.f);
    points_.resize(q_);
    for (int i = 0; i < q_; ++i) points_[i].slot = i;
  }

  // Start tracking a point (original image pixels) from the NEXT step() call.
  // Returns the slot id, or -1 if all slots are busy.
  int addPoint(float x, float y) {
    for (int i = 0; i < q_; ++i) {
      if (points_[i].active) continue;
      queries_[i * 3 + 0] = float(step_);  // defined on the frame about to run
      queries_[i * 3 + 1] = (x - roi_.x) * kImg / float(roi_.width);
      queries_[i * 3 + 2] = (y - roi_.y) * kImg / float(roi_.height);
      // A slot that was used before carries old history: wipe it.
      point_reset_[i] = used_[i] ? 1.f : 0.f;
      used_[i] = true;
      points_[i] = Point{};
      points_[i].slot = i;
      points_[i].active = true;
      points_[i].x = x;
      points_[i].y = y;
      fresh_[i] = true;
      return i;
    }
    return -1;
  }

  void removePoint(int slot) {
    points_[slot].active = false;
    queries_[slot * 3] = kInactiveT;
  }

  // Runs the model on one frame (BGR, original resolution).
  const std::vector<Point>& step(const cv::Mat& bgr) {
    preprocess(bgr);
    const int cur = ping_, nxt = 1 - ping_;
    const int64_t n = kPatches + q_;
    std::array<int64_t, 4> fshape{1, kImg, kImg, 3};
    std::array<int64_t, 3> qshape{1, q_, 3};
    std::array<int64_t, 1> sshape{1};
    std::array<int64_t, 2> rshape{1, q_};
    std::array<int64_t, 3> lshape{kLayers, n, kWidth};
    std::array<int64_t, 4> cshape{kLayers, n, 3, kWidth};
    float stepf = float(step_), vreset = step_ == 0 ? 1.f : 0.f;

    Ort::IoBinding io(*session_);
    io.BindInput("frame", mk(frame_.data(), frame_.size(), fshape));
    io.BindInput("query_points", mk(queries_.data(), queries_.size(), qshape));
    io.BindInput("step", mk(&stepf, 1, sshape));
    io.BindInput("video_reset", mk(&vreset, 1, sshape));
    io.BindInput("point_reset", mk(point_reset_.data(), point_reset_.size(), rshape));
    io.BindInput("lru_state", mk(lru_[cur].data(), lru_[cur].size(), lshape));
    io.BindInput("conv_state", mk(conv_[cur].data(), conv_[cur].size(), cshape));
    std::array<int64_t, 3> tshape{1, q_, 2};
    io.BindOutput("tracks_xy", mk(tracks_.data(), tracks_.size(), tshape));
    io.BindOutput("visible_prob", mk(vis_.data(), vis_.size(), rshape));
    io.BindOutput("certainty", mk(cert_.data(), cert_.size(), rshape));
    io.BindOutput("lru_state_out", mk(lru_[nxt].data(), lru_[nxt].size(), lshape));
    io.BindOutput("conv_state_out", mk(conv_[nxt].data(), conv_[nxt].size(), cshape));
    session_->Run(Ort::RunOptions{nullptr}, io);
    ping_ = nxt;

    const float sx = roi_.width / float(kImg), sy = roi_.height / float(kImg);
    for (int i = 0; i < q_; ++i) {
      point_reset_[i] = 0.f;
      Point& p = points_[i];
      if (!p.active) continue;
      const float x = roi_.x + tracks_[i * 2] * sx;
      const float y = roi_.y + tracks_[i * 2 + 1] * sy;
      if (!fresh_[i]) {
        p.vx = x - p.x;  // px/frame; smooth (EMA) if needed
        p.vy = y - p.y;
      }
      fresh_[i] = false;
      p.x = x;
      p.y = y;
      p.visible = vis_[i];
      p.certainty = cert_[i];
    }
    ++step_;
    return points_;
  }

  // Median velocity of reliable points inside a box: a robust motion prior
  // for anything rigidly attached to that vehicle (e.g. faces behind glass).
  bool medianVelocity(const cv::Rect2f& box, float& vx, float& vy) const {
    std::vector<float> xs, ys;
    for (const auto& p : points_) {
      if (p.active && p.reliable() && box.contains({p.x, p.y})) {
        xs.push_back(p.vx);
        ys.push_back(p.vy);
      }
    }
    if (xs.empty()) return false;
    auto med = [](std::vector<float>& v) {
      std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
      return v[v.size() / 2];
    };
    vx = med(xs);
    vy = med(ys);
    return true;
  }

 private:
  template <size_t R>
  Ort::Value mk(float* data, size_t count, const std::array<int64_t, R>& shape) {
    return Ort::Value::CreateTensor<float>(mem_, data, count, shape.data(), R);
  }

  void preprocess(const cv::Mat& bgr) {
    cv::Mat crop = bgr(roi_), rs, rgb;
    cv::resize(crop, rs, cv::Size(kImg, kImg), 0, 0, cv::INTER_AREA);
    cv::cvtColor(rs, rgb, cv::COLOR_BGR2RGB);
    cv::Mat f(kImg, kImg, CV_32FC3, frame_.data());
    rgb.convertTo(f, CV_32FC3, 2.0 / 255.0, -1.0);  // [-1, 1], HWC
  }

  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  int q_;
  cv::Rect roi_;
  Ort::MemoryInfo mem_;
  int64_t step_ = 0;
  int ping_ = 0;
  std::vector<float> frame_, queries_, point_reset_, tracks_, vis_, cert_;
  std::array<std::vector<float>, 2> lru_, conv_;
  std::vector<Point> points_;
  std::vector<char> used_ = std::vector<char>(q_, 0);
  std::vector<char> fresh_ = std::vector<char>(q_, 0);
};
