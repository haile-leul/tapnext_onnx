// Backend-independent part of the TAPNext C++ wrappers: point slots, ROI
// mapping, preprocessing and per-point velocities (original image pixels).
// Backends (tapnext_tracker.hpp = ONNX Runtime, tapnext_trt_tracker.hpp =
// TensorRT) only implement runModel(), which executes one step of the graph
// and keeps the recurrent state.
//
// Typical use (vehicle-assisted face velocity init):
//   for each frame:
//     for each newly confirmed vehicle track: tap.addPoint(x, y) for a few
//         points inside the vehicle box (e.g. a 3x3 grid around the windshield)
//     auto pts = tap.step(frame);
//     -> pts[i].vx, pts[i].vy are px/frame; medianVelocity(box) gives the
//        motion to seed a new face track's Kalman state.
#pragma once

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

class TapNextPoints {
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
  TapNextPoints(int num_slots, cv::Rect roi)
      : q_(num_slots),
        roi_(roi),
        frame_(kImg * kImg * 3, 0.f),
        queries_(num_slots * 3, 0.f),
        point_reset_(num_slots, 0.f),
        tracks_(num_slots * 2, 0.f),
        vis_(num_slots, 0.f),
        cert_(num_slots, 0.f),
        points_(num_slots),
        used_(num_slots, 0),
        fresh_(num_slots, 0) {
    for (int i = 0; i < q_; ++i) {
      queries_[i * 3] = kInactiveT;
      points_[i].slot = i;
    }
  }
  virtual ~TapNextPoints() = default;

  int numSlots() const { return q_; }
  int64_t frameIndex() const { return step_; }
  size_t lruStateSize() const { return size_t(kLayers) * (kPatches + q_) * kWidth; }
  size_t convStateSize() const { return lruStateSize() * 3; }

  // Start tracking a point (original image pixels) from the NEXT step() call.
  // Returns the slot id, or -1 if all slots are busy.
  int addPoint(float x, float y) {
    int slot = -1;
    for (int i = 0; i < q_ && slot < 0; ++i)  // prefer never-used slots
      if (!points_[i].active && !used_[i]) slot = i;
    for (int i = 0; i < q_ && slot < 0; ++i)
      if (!points_[i].active) slot = i;
    if (slot < 0) return -1;
    queries_[slot * 3 + 0] = float(step_);  // defined on the frame about to run
    queries_[slot * 3 + 1] = (x - roi_.x) * kImg / float(roi_.width);
    queries_[slot * 3 + 2] = (y - roi_.y) * kImg / float(roi_.height);
    // A slot that was used before carries old history: wipe it.
    point_reset_[slot] = used_[slot] ? 1.f : 0.f;
    used_[slot] = 1;
    fresh_[slot] = 1;
    points_[slot] = Point{};
    points_[slot].slot = slot;
    points_[slot].active = true;
    points_[slot].x = x;
    points_[slot].y = y;
    return slot;
  }

  void removePoint(int slot) {
    points_[slot].active = false;
    queries_[slot * 3] = kInactiveT;
  }

  // Runs the model on one frame (BGR, original resolution).
  const std::vector<Point>& step(const cv::Mat& bgr) {
    preprocess(bgr);
    runModel(float(step_), step_ == 0 ? 1.f : 0.f);
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
      fresh_[i] = 0;
      p.x = x;
      p.y = y;
      p.visible = vis_[i];
      p.certainty = cert_[i];
    }
    ++step_;
    return points_;
  }

  const std::vector<Point>& points() const { return points_; }

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

 protected:
  // Run one model step. Inputs: frame_, queries_, point_reset_ (host, fp32)
  // plus step / video_reset. Must fill tracks_ (Q x (x,y) in 256-px coords),
  // vis_, cert_ and advance the recurrent state.
  virtual void runModel(float step, float video_reset) = 0;

  void preprocess(const cv::Mat& bgr) {
    cv::Mat crop = bgr(roi_), rs, rgb;
    cv::resize(crop, rs, cv::Size(kImg, kImg), 0, 0, cv::INTER_AREA);
    cv::cvtColor(rs, rgb, cv::COLOR_BGR2RGB);
    cv::Mat f(kImg, kImg, CV_32FC3, frame_.data());
    rgb.convertTo(f, CV_32FC3, 2.0 / 255.0, -1.0);  // [-1, 1], HWC
  }

  int q_;
  cv::Rect roi_;
  int64_t step_ = 0;
  std::vector<float> frame_, queries_, point_reset_, tracks_, vis_, cert_;
  std::vector<Point> points_;
  std::vector<char> used_, fresh_;
};
