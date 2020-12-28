// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

constexpr int kNumFaceLandmarks = 40;
constexpr int kFaceLandmarksLocations[] = {
  // top outer lip
  61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 
  // bottom outer lip
  409, 270, 269, 267, 0, 37, 39, 40, 185,
  // 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 
  
  // bottom inner lip
  78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 
  // top inner lip
  415, 310, 311, 312, 13, 82, 81, 80, 191, 78
  // 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
};
constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kMaskTag[] = "MASK";
//  0,13,14,17,37,39,40,61,78,80,81,82,84,87,88,91,95,146,178,181,185,191,
//  267,269,270,291,308,310,311,312,314,317,318,321,324,375,402,405,409,415

Detection ConvertLandmarksToDetection(const NormalizedLandmarkList& landmarks) {
  Detection detection;
  LocationData* location_data = detection.mutable_location_data();

  float x_min = std::numeric_limits<float>::max();
  float x_max = std::numeric_limits<float>::min();
  float y_min = std::numeric_limits<float>::max();
  float y_max = std::numeric_limits<float>::min();
  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    const NormalizedLandmark& landmark = landmarks.landmark(i);
    x_min = std::min(x_min, landmark.x());
    x_max = std::max(x_max, landmark.x());
    y_min = std::min(y_min, landmark.y());
    y_max = std::max(y_max, landmark.y());

    auto keypoint = location_data->add_relative_keypoints();
    keypoint->set_x(landmark.x());
    keypoint->set_y(landmark.y());
  }

  location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  LocationData::RelativeBoundingBox* relative_bbox =
      location_data->mutable_relative_bounding_box();

  relative_bbox->set_xmin(x_min);
  relative_bbox->set_ymin(y_min);
  relative_bbox->set_width(x_max - x_min);
  relative_bbox->set_height(y_max - y_min);

  return detection;
}

float clip(float n, float lower, float upper) {
  return std::max(lower, std::min(n, upper));
}

// Produces result as an RGBA image, with the mask in both R & A channels. The
// value of each pixel is the probability of the specified class after softmax,
// scaled to 255 on CPU.
class FaceLandmarksToSegmentationCalculator : public CalculatorBase {
  public:
    //FaceLandmarksToSegmentationCalculator() = default;
    ~FaceLandmarksToSegmentationCalculator() override = default;

    static mediapipe::Status GetContract(CalculatorContract* cc);

    mediapipe::Status Open(CalculatorContext* cc) override;
    mediapipe::Status Process(CalculatorContext* cc) override;
};

REGISTER_CALCULATOR(FaceLandmarksToSegmentationCalculator);

// static
mediapipe::Status FaceLandmarksToSegmentationCalculator::GetContract (CalculatorContract *cc) {
  // Inputs CPU.
  cc->Inputs().Tag(kNormLandmarksTag).Set<NormalizedLandmarkList>();
  cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();

  // Outputs.
  cc->Outputs().Tag(kMaskTag).Set<ImageFrame>();
  return mediapipe::OkStatus();
}

mediapipe::Status FaceLandmarksToSegmentationCalculator::Open(CalculatorContext* cc) {
  return ::mediapipe::OkStatus();
}

mediapipe::Status FaceLandmarksToSegmentationCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().Tag(kNormLandmarksTag).IsEmpty()) {
    return mediapipe::OkStatus();
  }

  // Get input streams.
  const auto& input_landmarks =
      cc->Inputs().Tag(kNormLandmarksTag).Get<NormalizedLandmarkList>();
  const auto& image_size = cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();

  // Retrieve input properties.
  const int input_width = image_size.first;
  const int input_height = image_size.second;
  // std::cout << "Image Size [Width]: " << input_width << std::endl;
  // std::cout << "Image Size [Height]: " << input_height << std::endl;

  // Get the subset of normalized landmark indicies corresponding to the mouth.
  NormalizedLandmarkList subset_landmarks;
  std::cout << input_landmarks.landmark_size() << std::endl;
  for (int i = 0; i < kNumFaceLandmarks; ++i) {
    int landmark_location = kFaceLandmarksLocations[i];
    const NormalizedLandmark& landmark = input_landmarks.landmark(landmark_location);
    *subset_landmarks.add_landmark() = landmark;

    const int x = landmark.x() * input_width;
    const int y = landmark.y() * input_height;
    std::cout << "Landmark [" << i << "]: " << x << "," << y << std::endl;
  }

  // Find the detection box around the subset of landmarks.
  Detection detection = ConvertLandmarksToDetection(subset_landmarks);
  int xmin = input_width * detection.location_data().relative_bounding_box().xmin();
  int xmax = input_width * (
    detection.location_data().relative_bounding_box().xmin() + 
    detection.location_data().relative_bounding_box().width()
  );
  int ymin = input_height * detection.location_data().relative_bounding_box().ymin();
  int ymax = input_height * (
    detection.location_data().relative_bounding_box().ymin() +
    detection.location_data().relative_bounding_box().height()
  );
  std::cout << xmin << "," << xmax << "|" << ymin << "," << ymax << std::endl;
  xmin = clip(xmin, 0, input_width);
  xmax = clip(xmax, 0, input_width);
  ymin = clip(ymin, 0, input_height);
  ymax = clip(ymax, 0, input_height);

  cv::Mat mask_mat = cv::Mat::zeros(cv::Size(input_width, input_height), CV_8UC4);
  // mask_mat.setTo(cv::Scalar(255, 255, 255));
  // Iterate over the bounding box points as candidates.
  for (int test_y = ymin; test_y <= ymax; ++test_y) {
    for (int test_x = xmin; test_x <= xmax; ++test_x) {
      // Iterate over each landmark (polygon) vertex point.
      bool inside_polygon = false;
      int i, j = 0;
      for (int i = 0, j = kNumFaceLandmarks - 1; i < kNumFaceLandmarks; j = i++) {
        const NormalizedLandmark& landmark_i = subset_landmarks.landmark(i);
        const int vertex_xi = landmark_i.x() * input_width;
        const int vertex_yi = landmark_i.y() * input_height;

        const NormalizedLandmark& landmark_j = subset_landmarks.landmark(j);
        const int vertex_xj = landmark_j.x() * input_width;
        const int vertex_yj = landmark_j.y() * input_height;

        // Perform intersection count check.
        if (((vertex_yi > test_y) != (vertex_yj > test_y)) &&
            (test_x < (vertex_xj-vertex_xi) * (test_y-vertex_yi) / (vertex_yj-vertex_yi) + vertex_xi))
          inside_polygon = !inside_polygon;
      }

      // Set the points inside the polygon.
      const int mask_value = 255;
      if (inside_polygon) {
        // Set both R and A channels for convenience.
        const cv::Vec4b out_value = {mask_value, 0, 0, mask_value};
        std::cout << test_x << "," << test_y << std::endl;
        mask_mat.at<cv::Vec4b>(test_y, test_x) = out_value;
        std::cout << test_x << "," << test_y << std::endl;
      }
    }
  }
  
  // Send out image as CPU packet.
  std::unique_ptr<ImageFrame> output_mask = absl::make_unique<ImageFrame>(
      ImageFormat::SRGBA, input_width, input_height);
  cv::Mat output_mat = formats::MatView(output_mask.get());
  mask_mat.copyTo(output_mat);
  cc->Outputs().Tag(kMaskTag).Add(output_mask.release(), cc->InputTimestamp());

  return mediapipe::OkStatus();
}

}  // namespace mediapipe
