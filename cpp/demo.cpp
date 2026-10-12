// Usage: tapnext_demo <model.onnx | model.engine> <video or image pattern, e.g. frames/%03d.png> [cuda]
// .engine -> TensorRT backend (needs -DWITH_TENSORRT=ON), .onnx -> ONNX Runtime
// (CPU, or CUDA EP with the 'cuda' flag). Seeds a 4x4 grid of points in the
// frame centre on frame 0 and prints per-frame positions / velocities.
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>

#include "tapnext_tracker.hpp"
#ifdef TAPNEXT_WITH_TRT
#include "tapnext_trt_tracker.hpp"
#endif

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s model.{onnx,engine} video [cuda]\n", argv[0]);
    return 1;
  }
  cv::VideoCapture cap(argv[2]);
  cv::Mat frame;
  if (!cap.read(frame)) return 1;
  const cv::Rect roi(0, 0, frame.cols, frame.rows);  // whole frame; use a lane ROI in practice
  const std::string model = argv[1];
  std::unique_ptr<TapNextPoints> tap;
  if (model.size() > 7 && model.substr(model.size() - 7) == ".engine") {
#ifdef TAPNEXT_WITH_TRT
    tap = std::make_unique<TapNextTrtTracker>(model, /*num_slots=*/32, roi);
#else
    std::fprintf(stderr, "built without TensorRT (-DWITH_TENSORRT=ON)\n");
    return 1;
#endif
  } else {
    tap = std::make_unique<TapNextTracker>(model, /*num_slots=*/32, roi, argc > 3);
  }

  for (int t = 0; !frame.empty(); ++t) {
    if (t == 0) {
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
          tap->addPoint(frame.cols * (0.1f + 0.1f * j), frame.rows * (0.35f + 0.1f * i));
    }
    auto t0 = std::chrono::steady_clock::now();
    const auto& pts = tap->step(frame);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    float vx = 0, vy = 0;
    bool ok = tap->medianVelocity(cv::Rect2f(0, 0, frame.cols, frame.rows), vx, vy);
    std::printf("frame %d (%.1f ms): p0=(%.1f, %.1f) v0=(%.1f, %.1f) vis=%.2f cert=%.2f | median v=(%.1f, %.1f)%s\n",
                t, ms, pts[0].x, pts[0].y, pts[0].vx, pts[0].vy, pts[0].visible, pts[0].certainty,
                vx, vy, ok ? "" : " (none)");
    cap.read(frame);
  }
  return 0;
}
