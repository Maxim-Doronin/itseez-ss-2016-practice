#include <iostream>
#include <string>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

#include "image_processing.hpp"

using namespace std;
using namespace cv;

const char* kAbout =
    "This is an empty application that can be treated as a template for your "
    "own doing-something-cool applications.";

const char* kOptions =
	"{ @image         | <none> | image to process            }"
	"{ v video        | <none> | video to process            }"
	"{ gray           |        | convert image to gray scale }"
	"{ median         |        | apply median filter         }"
	"{ edges          |        | detect edges                }"
	"{ pix            |        | pixelize                    }"
  "{ dist           |        | distance transform          }"
  "{ aver           |        | average filter              }"
  "{ conv           |        | convolution                 }"
  "{ roi            |        | enable ROI                  }"
  "{ h ? help usage |        | print help message          }";

struct MousePosition {
	bool is_selection_started;
	bool is_selection_finished;
	Point point_first;
	Point point_second;
} mouse;

static void OnMouse(int event, int x, int y, int, void*) {
	switch (event) {
	case EVENT_LBUTTONDOWN :
		mouse.is_selection_started = true;
		mouse.is_selection_finished = false;
		mouse.point_first = Point(x, y);
		break;
	case EVENT_LBUTTONUP :
		mouse.is_selection_started = true;
		mouse.is_selection_finished = true;
		mouse.point_second = Point(x, y);
		break;
	case EVENT_MOUSEMOVE :
		if (mouse.is_selection_started && !mouse.is_selection_finished)
			mouse.point_second = Point(x, y);
	}
}

const int medianSize = 5;
const int edgesFilterSize = 5;
const int edgesLowThreshold = 200;
const int edgesRatio = 3;
const int edgesKernelSize = 5;
const int pixDivs = 10;
const int distLowThreshold = 100;
const int distHighThreshold = 255;
const int distanceType = CV_DIST_L1;
const int distMaskSize = 3;

int main(int argc, const char** argv) {
  // Parse command line arguments.
  CommandLineParser parser(argc, argv, kOptions);
  parser.about(kAbout);

  // If help option is given, print help message and exit.
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }

  Mat src, dst;
  VideoCapture cap("C://Users//doro1//dev//itseez-ss-2016-practice-build//bin//lena.png");
  cap >> src;
  if (src.empty()) {
	  cout << "Failed to open image file '" + parser.get<string>(0) + "'."
	  << endl;
	  return 0;
  }

  const string kSrcWindowName = "Source image";
  const int kWaitKeyDelay = 1;
  namedWindow(kSrcWindowName, WINDOW_NORMAL);
  setMouseCallback(kSrcWindowName, OnMouse, 0);
  resizeWindow(kSrcWindowName, src.cols, src.rows);
  imshow(kSrcWindowName, src);
  waitKey(kWaitKeyDelay);

  Rect roi;
  if (!parser.get<bool>("roi")) {
      roi = Rect(0, 0, src.cols, src.rows);
  }
  else {
    mouse.is_selection_started = false;
    mouse.is_selection_finished = false;
    Rect rect;
    while (!mouse.is_selection_finished) {
      if (mouse.is_selection_started) {
        Mat src_cpy;
        src.copyTo(src_cpy);
        rect.x = mouse.point_first.x;
        rect.y = mouse.point_first.y;
        rect.width = mouse.point_second.x - mouse.point_first.x;
        rect.height = mouse.point_second.y - mouse.point_first.y;
        rectangle(src_cpy, rect, Scalar(254));
        imshow(kSrcWindowName, src_cpy);
      }
      waitKey(30);
    }

    roi = Rect(mouse.point_first.x, mouse.point_first.y,
      mouse.point_second.x - mouse.point_first.x,
      mouse.point_second.y - mouse.point_first.y);
  }

  ImageProcessorImpl proc;
  bool firstFrame = true;
  const string kDstWindowName = "Destination image";
  namedWindow(kDstWindowName, WINDOW_NORMAL);
  resizeWindow(kDstWindowName, 640, 480);
  for (;;) {
	  if (src.empty() && !firstFrame)
		  break;

	  if (parser.get<bool>("gray")) {
		  dst = proc.CvtColor(src, roi);
	  }

	  if (parser.get<bool>("median")) {
		  dst = proc.Filter(src, roi, medianSize);
	  }

	  if (parser.get<bool>("edges")) {
		  dst = proc.DetectEdges(src, roi, edgesFilterSize, edgesLowThreshold,
			  edgesRatio, edgesKernelSize);
	  }

	  if (parser.get<bool>("pix")) {
		  dst = proc.Pixelize(src, roi, pixDivs);
	  }

    if (parser.get<bool>("dist")) {
      dst = proc.DistanceTransform(src, distLowThreshold,
        distHighThreshold, distanceType, distMaskSize);
    }

    if (parser.get<bool>("aver")) {
      dst = proc.AverageFilter(src);
    }

    if (parser.get<bool>("conv")) {
      dst = proc.Convolution(src);
    }

	  firstFrame = false;
	  cap >> src;
    if (parser.get<bool>("roi")) {
      rectangle(dst, roi, Scalar(254));
    }
	  imshow(kDstWindowName, dst);
    resizeWindow(kDstWindowName, dst.cols, dst.rows);
	  if(waitKey(30) >= 0) break;
  }
  waitKey();
  return 0;
}
