// Track-On2 TensorRT demo:
//   trackon_demo <encoder.engine> <tracker.engine> <video> [out.mp4]
// Adds an 8x8 grid on frame 0, 5 more points at frame 30, removes every other
// point at frame 60 and drops points that stay invisible for 10 frames.
// Writes a video with the tracks and prints GPU timings.
#include <chrono>
#include <cstdio>
#include <map>

#include "trackon_trt.hpp"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s encoder.engine tracker.engine video [out.mp4]\n", argv[0]);
    return 1;
  }
  cv::VideoCapture cap(argv[3]);
  if (!cap.isOpened()) return 1;
  const int W = int(cap.get(cv::CAP_PROP_FRAME_WIDTH)), H = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  cv::VideoWriter out(argc > 4 ? argv[4] : "trackon_demo.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                      cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 25, cv::Size(W, H));
  TrackOnTrt tracker(argv[1], argv[2]);
  std::printf("engine supports up to %d points\n", tracker.maxPoints());

  std::map<int, std::vector<cv::Point>> trails;
  cv::Mat frame;
  double sumMs = 0, sumGpu = 0;
  int timed = 0;
  for (int t = 0; cap.read(frame); ++t) {
    if (t == 0)
      for (int j = 0; j < 8; ++j)
        for (int i = 0; i < 8; ++i) tracker.addPoint(W * (i + 0.5f) / 8, H * (j + 0.5f) / 8);
    if (t == 30)
      for (int i = 0; i < 5; ++i) tracker.addPoint(W * (0.3f + 0.1f * i), H * 0.5f);
    if (t == 60) {
      std::vector<int> ids;
      for (size_t i = 0; i < tracker.points().size(); i += 2) ids.push_back(tracker.points()[i].id);
      tracker.removePoints(ids);
    }
    for (const auto& p : tracker.points())
      if (p.lost >= 10) tracker.removePoint(p.id);

    auto t0 = std::chrono::steady_clock::now();
    const auto& pts = tracker.step(frame);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (t >= 5) sumMs += ms, sumGpu += tracker.encoderMs() + tracker.trackerMs(), ++timed;

    for (const auto& p : pts) {
      auto& tr = trails[p.id];
      tr.push_back(cv::Point(int(p.x), int(p.y)));
      if (tr.size() > 20) tr.erase(tr.begin());
      if (!p.visible) continue;
      cv::Scalar c((p.id * 67) % 255, (p.id * 131) % 255, 255 - (p.id * 41) % 255);
      for (size_t i = 1; i < tr.size(); ++i) cv::line(frame, tr[i - 1], tr[i], c, 2, cv::LINE_AA);
      cv::circle(frame, tr.back(), 4, c, -1, cv::LINE_AA);
    }
    char hud[160];
    std::snprintf(hud, sizeof(hud), "frame %d  %d points  step %.1f ms (GPU enc %.1f + trk %.1f)", t,
                  int(pts.size()), ms, tracker.encoderMs(), tracker.trackerMs());
    cv::putText(frame, hud, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 0, 0}, 4, cv::LINE_AA);
    cv::putText(frame, hud, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {255, 255, 255}, 2, cv::LINE_AA);
    out.write(frame);
    if (t % 25 == 0) std::printf("%s\n", hud);
  }
  if (timed) std::printf("mean step %.1f ms (GPU %.1f ms) -> %.1f fps\n", sumMs / timed, sumGpu / timed, 1000.0 * timed / sumMs);
  return 0;
}
