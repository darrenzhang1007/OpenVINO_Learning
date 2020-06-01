#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;


string model = "E:/models/tf_models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb";
string config = "E:/models/tf_models/ssd_mobilenet_v2_coco_2018_03_29/graph.pbtxt";
int main(int argc, char** argv) {
	Mat src = imread("H:/OpenVINO_Learning/openvino_001/dog.jpg");
	imshow("src", src);
	if (src.empty()) {
		printf("could not load img...");
	}
	// load model
	Net net = readNetFromTensorflow(model, config);
	Mat blob = blobFromImage(src, 1.0, Size(300, 300), Scalar(), true, false, 5);
	net.setInput(blob);

	// Ç°Ïò´«²¥
	float threshold = 0.5;
	Mat detection = net.forward();

	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	for (int i = 0; i < detectionMat.rows; i++) {
		float confidence = detectionMat.at<float>(i, 2);
		if (confidence > threshold) {
			size_t obj_index = (size_t)detectionMat.at<float>(i, 1);
			float tl_x = detectionMat.at<float>(i, 3) *  src.cols;
			float tl_y = detectionMat.at<float>(i, 4) *  src.rows;
			float br_x = detectionMat.at<float>(i, 5) *  src.cols;
			float br_y = detectionMat.at<float>(i, 6) *  src.rows;
			Rect object_box((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int(br_y - tl_y)));
			rectangle(src, object_box, Scalar(0, 0, 255), 2, 8, 0);
		}
	}

	imshow("ssd_detectron", src);
	waitKey(0);
	destroyAllWindows();
	return 0;
}