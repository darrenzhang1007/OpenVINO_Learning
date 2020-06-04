#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// pip install test-generator==0.1.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/
// pip install defusedxml==0.5.0  -i https://pypi.tuna.tsinghua.edu.cn/simple/
// python mo_tf.py --input_model E:\models\tf_models\ssd_mobilenet_v2_coco_2018_03_29\frozen_inference_graph.pb --tensorflow_use_custom_operations_config E:\models\tf_models\ssd_mobilenet_v2_coco_2018_03_29\ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config E:\models\tf_models\ssd_mobilenet_v2_coco_2018_03_29\pipeline.config    --data_type FP32

void dnn_tf();
void dnn_ir_tf();


int main(int argc, char** argv) {
	dnn_tf();
	waitKey(0);
	destroyAllWindows();
	return 0;
}


void dnn_tf() {
	string model = "E:/models/tf_models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb";
	string config = "E:/models/tf_models/ssd_mobilenet_v2_coco_2018_03_29/graph.pbtxt";

	Mat src = imread("H:/OpenVINO_Learning/001_SSD_DNN_Load/dog.jpg");
	imshow("src", src);
	if (src.empty()) {
		printf("could not load img...\n");
	}
	// load model
	Net net = readNetFromTensorflow(model, config);
	net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE); // IE 加速推断
															//net.setPreferableBackend(DNN_BACKEND_OPENCV); // net.setPreferableBackend(DNN_BACKEND_OPENCV); // 指定计算后台
	net.setPreferableTarget(DNN_TARGET_CPU);  // 指定在什么设备上运行
	printf("loading model...\n");
	Mat blob = blobFromImage(src, 1.0, Size(300, 300), Scalar(), true, false, 5);
	net.setInput(blob);

	// 前向传播
	float threshold = 0.5;
	Mat detection = net.forward();

	// 获取推断时间
	vector<double> layerTimings;
	double freq = getTickFrequency() / 1000; // 毫秒数
	double time = net.getPerfProfile(layerTimings) / freq;
	ostringstream ss;
	ss << "inference:" << time << "ms";
	putText(src, ss.str(), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2, 8);

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
}


void dnn_ir_tf() {
	string xml = "E:/models/tf_models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml";
	string bin = "E:/models/tf_models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.bin";
	Mat src = imread("H:/OpenVINO_Learning/openvino_001/dog.jpg");

	// load model
	Net net = readNetFromModelOptimizer(xml, bin);
	net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE); // IE 加速推断
	net.setPreferableTarget(DNN_TARGET_CPU);  // 指定在什么设备上运行
	printf("loading model...\n");
	Mat blob = blobFromImage(src, 1.0, Size(300, 300), Scalar(), true, false, 5);
	net.setInput(blob);

	// 前向传播
	float threshold = 0.5;
	Mat detection = net.forward();

	// 获取推断时间
	vector<double> layerTimings;
	double freq = getTickFrequency() / 1000; // 毫秒数
	double time = net.getPerfProfile(layerTimings) / freq;
	ostringstream ss;
	ss << "inference:" << time << "ms";
	putText(src, ss.str(), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2, 8);

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
	imshow("ssd_model_optimizer", src);
}