// Track-On2 point tracker on TensorRT (TensorRT 10 API, CUDA runtime, OpenCV).
//
//   TrackOnTrt tracker("trackon_encoder.engine", "trackon_tracker.engine");
//   int id = tracker.addPoint(x, y);          // original image pixels, starts next step()
//   tracker.removePoint(id);                  // any time
//   const auto& pts = tracker.step(frameBgr);  // all active points
//   for (auto& p : pts) { p.id, p.x, p.y, p.vx, p.vy, p.visProb, p.visible }
//   float vx, vy;  if (tracker.medianVelocity(vehicleTrack.pointIds, vx, vy)) ...
//
// Per-point state (query feature, 72-frame memory, memory mask) lives on the GPU
// in two ping-pong buffers. Removing points compacts the rows on the GPU, adding
// points appends rows; nothing else is copied between frames except the
// 512x384 input frame and the N output points. The number of points is limited
// only by the tracker engine's profile (tools/build_trt_engines.py --max-points).
#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class TrackOnTrt {
 public:
  static constexpr int kInH = 384, kInW = 512;  // model input size

  struct Point {
    int id = -1;
    float x = 0, y = 0;    // original image pixels
    float vx = 0, vy = 0;  // pixels / frame (0 on the frame a point is added)
    float visProb = 0;     // sigmoid(visibility logit)
    bool visible = false;  // visProb >= deltaV (0.8, as in the official Predictor)
    int lost = 0;          // consecutive frames not visible
    bool hasVelocity = false;  // false on the frame the point was added (vx = vy = 0)
  };

  // roi: part of the frame given to the model (empty = whole frame). It is
  // resized to 512x384 like the official Predictor resizes the full frame.
  TrackOnTrt(const std::string& encoderEngine, const std::string& trackerEngine,
             cv::Rect roi = cv::Rect(), float deltaV = 0.8f, cudaStream_t stream = nullptr)
      : roi_(roi), deltaV_(deltaV), stream_(stream) {
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    enc_.reset(load(encoderEngine));
    trk_.reset(load(trackerEngine));
    encCtx_.reset(enc_->createExecutionContext());
    trkCtx_.reset(trk_->createExecutionContext());
    if (!stream_) {
      check(cudaStreamCreate(&stream_));
      ownStream_ = true;
    }
    for (auto& e : ev_) check(cudaEventCreate(&e));

    // encoder: static shapes
    auto fdims = enc_->getTensorShape("frame");
    check(hostAlloc(&hFrame_, volume(fdims) * sizeof(float)));
    dFrame_ = alloc(volume(fdims));
    setShape(encCtx_.get(), "frame", fdims);
    bind(encCtx_.get(), "frame", dFrame_);
    for (int i = 0; i < 5; ++i) {
      auto d = enc_->getTensorShape(kFeat[i]);
      dFeat_[i] = alloc(volume(d));
      bind(encCtx_.get(), kFeat[i], dFeat_[i]);
      setShape(trkCtx_.get(), kFeat[i], d);
      bind(trkCtx_.get(), kFeat[i], dFeat_[i]);
    }
    // tracker: dynamic number of points
    D_ = int(trk_->getTensorShape("q_init").d[1]);
    M_ = int(trk_->getTensorShape("memory").d[1]);
    minN_ = int(trk_->getProfileShape("q_init", 0, nvinfer1::OptProfileSelector::kMIN).d[0]);
    maxN_ = int(trk_->getProfileShape("q_init", 0, nvinfer1::OptProfileSelector::kMAX).d[0]);
    reserve(64);
  }

  ~TrackOnTrt() {
    for (void* p : dAllocs_) cudaFree(p);
    freeRows();
    if (hFrame_) cudaFreeHost(hFrame_);
    for (auto& e : ev_) cudaEventDestroy(e);
    if (ownStream_) cudaStreamDestroy(stream_);
  }

  // Start tracking a point (original image pixels) from the next step(). Returns its id.
  int addPoint(float x, float y) {
    pending_.push_back({nextId_, x, y});
    return nextId_++;
  }
  void removePoint(int id) { removeIds_.insert(id); }
  void removePoints(const std::vector<int>& ids) { removeIds_.insert(ids.begin(), ids.end()); }
  int numPoints() const { return n_ + int(pending_.size()) - pendingRemovalsOfPending(); }
  int maxPoints() const { return maxN_; }
  const std::vector<Point>& points() const { return out_; }
  float encoderMs() const { return encMs_; }  // GPU time of the last step
  float trackerMs() const { return trkMs_; }

  // Median velocity (px/frame) of the given point ids -- e.g. the ids you sampled
  // from one vehicle detection and stored in its track. Uses only points that
  // are visible and already have a velocity (not on the frame they were added).
  // Returns the number of points used; 0 = no estimate (vx, vy untouched).
  //   speed [px/s] = hypot(vx, vy) * fps,  direction = atan2(vy, vx)
  int medianVelocity(const std::vector<int>& ids, float& vx, float& vy) const {
    std::unordered_set<int> want(ids.begin(), ids.end());
    return medianOf([&](const Point& p) { return want.count(p.id) > 0; }, vx, vy);
  }

  // Same, for the points currently inside a box (original px). Prefer the id
  // version: a box can also contain points of an overlapping object.
  int medianVelocity(const cv::Rect2f& box, float& vx, float& vy) const {
    return medianOf([&](const Point& p) { return box.contains({p.x, p.y}); }, vx, vy);
  }

  const std::vector<Point>& step(const cv::Mat& bgr) {
    applyRemovals();
    const int k = appendPending(bgr.cols, bgr.rows);
    if (n_ == 0) {
      out_.clear();
      return out_;
    }
    if (n_ < minN_) throw std::runtime_error("fewer points than the engine profile minimum");
    preprocess(bgr);

    check(cudaMemcpyAsync(dFrame_, hFrame_, size_t(3) * kInH * kInW * sizeof(float),
                          cudaMemcpyHostToDevice, stream_));
    check(cudaEventRecord(ev_[0], stream_));
    if (!encCtx_->enqueueV3(stream_)) throw std::runtime_error("encoder enqueueV3 failed");
    check(cudaEventRecord(ev_[1], stream_));

    // new-point flags/positions for the tracker (only the last k rows are new)
    std::fill(hIsNew_, hIsNew_ + n_, 0.f);
    for (int i = n_ - k; i < n_; ++i) hIsNew_[i] = 1.f;
    check(cudaMemcpyAsync(dXy_, hXy_, size_t(n_) * 2 * sizeof(float), cudaMemcpyHostToDevice, stream_));
    check(cudaMemcpyAsync(dIsNew_, hIsNew_, size_t(n_) * sizeof(float), cudaMemcpyHostToDevice, stream_));
    auto* c = trkCtx_.get();
    setShape(c, "q_init", dims({n_, D_}));
    setShape(c, "memory", dims({n_, M_, D_}));
    setShape(c, "memory_mask", dims({n_, M_}));
    setShape(c, "new_xy", dims({n_, 2}));
    setShape(c, "is_new", dims({n_}));
    const int p = ping_, q = 1 - ping_;
    bind(c, "q_init", dQ_[p]);
    bind(c, "memory", dMem_[p]);
    bind(c, "memory_mask", dMask_[p]);
    bind(c, "new_xy", dXy_);
    bind(c, "is_new", dIsNew_);
    bind(c, "points", dPts_);
    bind(c, "vis_logit", dVis_);
    bind(c, "unc_logit", dUnc_);
    bind(c, "q_init_out", dQ_[q]);
    bind(c, "memory_out", dMem_[q]);
    bind(c, "memory_mask_out", dMask_[q]);
    if (!c->enqueueV3(stream_)) throw std::runtime_error("tracker enqueueV3 failed");
    check(cudaEventRecord(ev_[2], stream_));
    check(cudaMemcpyAsync(hPts_, dPts_, size_t(n_) * 2 * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    check(cudaMemcpyAsync(hVis_, dVis_, size_t(n_) * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    check(cudaStreamSynchronize(stream_));
    check(cudaEventElapsedTime(&encMs_, ev_[0], ev_[1]));
    check(cudaEventElapsedTime(&trkMs_, ev_[1], ev_[2]));
    ping_ = q;

    // outputs in original pixels + velocities
    const float sx = roiW_ / float(kInW), sy = roiH_ / float(kInH);
    out_.resize(n_);
    for (int r = 0; r < n_; ++r) {
      Point pt;
      pt.id = rowIds_[r];
      pt.x = roiX_ + hPts_[2 * r] * sx;
      pt.y = roiY_ + hPts_[2 * r + 1] * sy;
      pt.visProb = 1.f / (1.f + std::exp(-hVis_[r]));
      pt.visible = pt.visProb >= deltaV_;
      auto it = prev_.find(pt.id);
      if (it != prev_.end() && r < n_ - k) {
        pt.vx = pt.x - it->second.x;
        pt.vy = pt.y - it->second.y;
        pt.hasVelocity = true;
        pt.lost = pt.visible ? 0 : it->second.lost + 1;
      } else {
        pt.lost = pt.visible ? 0 : 1;
      }
      out_[r] = pt;
    }
    prev_.clear();
    for (const auto& pt : out_) prev_[pt.id] = pt;
    return out_;
  }

 private:
  static constexpr const char* kFeat[5] = {"f4", "f8", "f16", "f32", "f_fused"};
  struct Pending { int id; float x, y; };
  struct Logger : nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
      if (s <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
    }
  };
  struct Del {
    template <class T>
    void operator()(T* p) const { delete p; }
  };

  template <class Pred>
  int medianOf(Pred use, float& vx, float& vy) const {
    std::vector<float> xs, ys;
    for (const auto& p : out_)
      if (p.visible && p.hasVelocity && use(p)) xs.push_back(p.vx), ys.push_back(p.vy);
    if (xs.empty()) return 0;
    auto med = [](std::vector<float>& v) {
      std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
      return v[v.size() / 2];
    };
    vx = med(xs);
    vy = med(ys);
    return int(xs.size());
  }

  static void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
  }
  template <class T>
  static cudaError_t devAlloc(T** p, size_t bytes) { return cudaMalloc(reinterpret_cast<void**>(p), bytes); }
  template <class T>
  static cudaError_t hostAlloc(T** p, size_t bytes) { return cudaMallocHost(reinterpret_cast<void**>(p), bytes); }
  static size_t volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= size_t(d.d[i]);
    return v;
  }
  static nvinfer1::Dims dims(std::initializer_list<int64_t> l) {
    nvinfer1::Dims d{};
    d.nbDims = int(l.size());
    int i = 0;
    for (auto v : l) d.d[i++] = v;
    return d;
  }
  nvinfer1::ICudaEngine* load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto* e = runtime_->deserializeCudaEngine(blob.data(), blob.size());
    if (!e) throw std::runtime_error(path + ": deserialization failed (other TensorRT version / GPU?)");
    return e;
  }
  float* alloc(size_t count) {
    void* p = nullptr;
    check(cudaMalloc(&p, count * sizeof(float)));
    dAllocs_.push_back(p);
    return static_cast<float*>(p);
  }
  static void setShape(nvinfer1::IExecutionContext* c, const char* name, const nvinfer1::Dims& d) {
    if (!c->setInputShape(name, d))
      throw std::runtime_error(std::string("shape not allowed by the engine profile: ") + name);
  }
  static void bind(nvinfer1::IExecutionContext* c, const char* name, void* p) {
    if (!c->setTensorAddress(name, p)) throw std::runtime_error(std::string("bad tensor ") + name);
  }

  // ---- per-point buffers (capacity grows x2) ----
  void freeRows() {
    for (int i = 0; i < 2; ++i) {
      if (dQ_[i]) cudaFree(dQ_[i]), cudaFree(dMem_[i]), cudaFree(dMask_[i]);
      dQ_[i] = dMem_[i] = dMask_[i] = nullptr;
    }
    for (float* p : {dXy_, dIsNew_, dPts_, dVis_, dUnc_}) if (p) cudaFree(p);
    for (float* p : {hXy_, hIsNew_, hPts_, hVis_}) if (p) cudaFreeHost(p);
    dXy_ = dIsNew_ = dPts_ = dVis_ = dUnc_ = hXy_ = hIsNew_ = hPts_ = hVis_ = nullptr;
  }
  void reserve(int cap) {
    if (cap <= cap_) return;
    float *q[2], *mem[2], *mask[2];
    for (int i = 0; i < 2; ++i) {
      check(devAlloc(&q[i], size_t(cap) * D_ * sizeof(float)));
      check(devAlloc(&mem[i], size_t(cap) * M_ * D_ * sizeof(float)));
      check(devAlloc(&mask[i], size_t(cap) * M_ * sizeof(float)));
    }
    if (n_) {  // keep current rows
      check(cudaMemcpyAsync(q[0], dQ_[ping_], size_t(n_) * D_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
      check(cudaMemcpyAsync(mem[0], dMem_[ping_], size_t(n_) * M_ * D_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
      check(cudaMemcpyAsync(mask[0], dMask_[ping_], size_t(n_) * M_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
      check(cudaStreamSynchronize(stream_));
    }
    std::vector<float> keepXy(hXy_ ? hXy_ : nullptr, hXy_ ? hXy_ + 2 * size_t(n_) : nullptr);
    freeRows();
    for (int i = 0; i < 2; ++i) dQ_[i] = q[i], dMem_[i] = mem[i], dMask_[i] = mask[i];
    ping_ = 0;
    check(devAlloc(&dXy_, size_t(cap) * 2 * sizeof(float)));
    check(devAlloc(&dIsNew_, size_t(cap) * sizeof(float)));
    check(devAlloc(&dPts_, size_t(cap) * 2 * sizeof(float)));
    check(devAlloc(&dVis_, size_t(cap) * sizeof(float)));
    check(devAlloc(&dUnc_, size_t(cap) * sizeof(float)));
    check(hostAlloc(&hXy_, size_t(cap) * 2 * sizeof(float)));
    check(hostAlloc(&hIsNew_, size_t(cap) * sizeof(float)));
    check(hostAlloc(&hPts_, size_t(cap) * 2 * sizeof(float)));
    check(hostAlloc(&hVis_, size_t(cap) * sizeof(float)));
    std::copy(keepXy.begin(), keepXy.end(), hXy_);
    cap_ = cap;
  }

  int pendingRemovalsOfPending() const {
    int c = 0;
    for (const auto& p : pending_) c += removeIds_.count(p.id) ? 1 : 0;
    return c;
  }

  // Drop removed rows: copy the kept runs into the other buffer, then swap.
  void applyRemovals() {
    if (removeIds_.empty()) return;
    pending_.erase(std::remove_if(pending_.begin(), pending_.end(),
                                  [&](const Pending& p) { return removeIds_.count(p.id) > 0; }),
                   pending_.end());
    std::vector<int> keep;
    for (int r = 0; r < n_; ++r)
      if (!removeIds_.count(rowIds_[r])) keep.push_back(r);
    removeIds_.clear();
    if (int(keep.size()) == n_) return;
    const int src = ping_, dst = 1 - ping_;
    std::vector<int> ids;
    for (size_t i = 0; i < keep.size();) {  // contiguous runs
      size_t j = i;
      while (j + 1 < keep.size() && keep[j + 1] == keep[j] + 1) ++j;
      const size_t len = j - i + 1, from = size_t(keep[i]), to = i;
      check(cudaMemcpyAsync(dQ_[dst] + to * D_, dQ_[src] + from * D_, len * D_ * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream_));
      check(cudaMemcpyAsync(dMem_[dst] + to * M_ * D_, dMem_[src] + from * M_ * D_,
                            len * M_ * D_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
      check(cudaMemcpyAsync(dMask_[dst] + to * M_, dMask_[src] + from * M_, len * M_ * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream_));
      i = j + 1;
    }
    for (int r : keep) ids.push_back(rowIds_[r]);
    rowIds_ = ids;
    n_ = int(keep.size());
    ping_ = dst;
  }

  // Append pending points as new rows (zero query/memory, fully masked memory).
  int appendPending(int frameW, int frameH) {
    roiX_ = roi_.width > 0 ? float(roi_.x) : 0.f;
    roiY_ = roi_.width > 0 ? float(roi_.y) : 0.f;
    roiW_ = roi_.width > 0 ? float(roi_.width) : float(frameW);
    roiH_ = roi_.width > 0 ? float(roi_.height) : float(frameH);
    const int k = int(pending_.size());
    if (k == 0) return 0;
    if (n_ + k > maxN_)
      throw std::runtime_error("too many points for the engine (rebuild with a larger --max-points)");
    if (n_ + k > cap_) reserve(std::max(n_ + k, 2 * cap_));
    const int p = ping_;
    check(cudaMemsetAsync(dQ_[p] + size_t(n_) * D_, 0, size_t(k) * D_ * sizeof(float), stream_));
    check(cudaMemsetAsync(dMem_[p] + size_t(n_) * M_ * D_, 0, size_t(k) * M_ * D_ * sizeof(float), stream_));
    if (ones_.size() < size_t(k) * M_) ones_.assign(size_t(k) * M_, 1.f);
    check(cudaMemcpyAsync(dMask_[p] + size_t(n_) * M_, ones_.data(), size_t(k) * M_ * sizeof(float),
                          cudaMemcpyHostToDevice, stream_));
    for (int i = 0; i < k; ++i) {
      hXy_[2 * (n_ + i)] = (pending_[i].x - roiX_) * kInW / roiW_;
      hXy_[2 * (n_ + i) + 1] = (pending_[i].y - roiY_) * kInH / roiH_;
      rowIds_.push_back(pending_[i].id);
    }
    check(cudaStreamSynchronize(stream_));  // ones_ is pageable host memory
    pending_.clear();
    n_ += k;
    return k;
  }

  void preprocess(const cv::Mat& bgr) {
    cv::Mat src = roi_.width > 0 ? bgr(roi_) : bgr, rs, rgb;
    cv::resize(src, rs, cv::Size(kInW, kInH), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(rs, rgb, cv::COLOR_BGR2RGB);
    std::vector<cv::Mat> planes = {
        cv::Mat(kInH, kInW, CV_32F, hFrame_),
        cv::Mat(kInH, kInW, CV_32F, hFrame_ + kInH * kInW),
        cv::Mat(kInH, kInW, CV_32F, hFrame_ + 2 * kInH * kInW)};
    cv::Mat f;
    rgb.convertTo(f, CV_32FC3);  // 0..255, the engine divides by 255
    cv::split(f, planes);        // HWC -> CHW straight into the pinned buffer
  }

  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime, Del> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, Del> enc_, trk_;
  std::unique_ptr<nvinfer1::IExecutionContext, Del> encCtx_, trkCtx_;
  cv::Rect roi_;
  float deltaV_;
  cudaStream_t stream_ = nullptr;
  bool ownStream_ = false;
  cudaEvent_t ev_[3]{};
  float encMs_ = 0, trkMs_ = 0;
  std::vector<void*> dAllocs_;
  float* hFrame_ = nullptr;
  float* dFrame_ = nullptr;
  float* dFeat_[5]{};
  int D_ = 0, M_ = 0, minN_ = 1, maxN_ = 1, n_ = 0, cap_ = 0, ping_ = 0, nextId_ = 0;
  float *dQ_[2]{}, *dMem_[2]{}, *dMask_[2]{};
  float *dXy_ = nullptr, *dIsNew_ = nullptr, *dPts_ = nullptr, *dVis_ = nullptr, *dUnc_ = nullptr;
  float *hXy_ = nullptr, *hIsNew_ = nullptr, *hPts_ = nullptr, *hVis_ = nullptr;
  float roiX_ = 0, roiY_ = 0, roiW_ = 1, roiH_ = 1;
  std::vector<float> ones_;
  std::vector<int> rowIds_;
  std::vector<Pending> pending_;
  std::unordered_set<int> removeIds_;
  std::unordered_map<int, Point> prev_;
  std::vector<Point> out_;
};
