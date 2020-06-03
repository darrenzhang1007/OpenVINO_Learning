#include <inference_engine.hpp>
#include <ext_list.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace InferenceEngine;
using namespace std;


int main(int argc, char** argv) {
	string bin = "E:/models/tf_models/vehicle-license-plate-detection-barrier-0106/vehicle-license-plate-detection-barrier-0106.bin";
	string xml = "E:/models/tf_models/vehicle-license-plate-detection-barrier-0106/vehicle-license-plate-detection-barrier-0106.xml";

	Mat src = imread("H:/OpenVINO_Learning/002_Vehicle_license_Plate_detection/vehicle_01.jpg");
	if (src.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input", WINDOW_NORMAL);
	namedWindow("ssd-model optimizer demo", WINDOW_NORMAL);
	imshow("input", src);
	int image_width = src.cols;
	int image_height = src.rows;

	vector<file_name_t> dirs;
	std::string s("C:/Program Files (x86)/IntelSWTools/openvino_2019.2.242/deployment_tools/inference_engine/bin/intel64/Debug");
	std::wstring ws;
	ws.assign(s.begin(), s.end());
	dirs.push_back(ws);
	
	// 创建IE插件
	// 创建IE插件, 查询支持硬件设备
	InferenceEngine::Core ie;
	vector<string> availableDevices = ie.GetAvailableDevices();
	for (int i = 0; i < availableDevices.size(); i++) {
		printf("supported device name : %s \n", availableDevices[i].c_str());
	}

	// 加载CPU扩展支持
	ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
	
	// 加载网络
	CNNNetReader network_reader;
	network_reader.ReadNetwork(xml);
	network_reader.ReadWeights(bin);

	// 获取输入输出
	auto network = network_reader.getNetwork();
	InferenceEngine::InputsDataMap input_info(network.getInputsInfo());
	InferenceEngine::OutputsDataMap output_info(network.getOutputsInfo());

	// 设置输入输出
	for (auto &item : input_info) {
		auto input_data = item.second;
		input_data->setPrecision(Precision::U8);
		input_data->setLayout(Layout::NCHW);
	}

	for (auto &item : output_info) {
		auto output_data = item.second;
		output_data->setPrecision(Precision::FP32);
	}
	
	// 创建可执行网络
	auto executable_network = ie.LoadNetwork(network, "CPU");

	// 请求推断
	auto infer_request = executable_network.CreateInferRequest();

	// 设置输入数据
	for (auto &item : input_info) {
		auto inut_name = item.first;
		auto input = infer_request.GetBlob(inut_name);
		size_t num_channels = input->getTensorDesc().getDims()[1];
		size_t h = input->getTensorDesc().getDims()[2];
		size_t w = input->getTensorDesc().getDims()[3];
		size_t image_size = w*h;
		Mat blob_image;
		resize(src, blob_image, Size(w, h));

		// NCHW
		unsigned char* data = static_cast<unsigned char*>(input->buffer());
		for (size_t row = 0; row < h; row++) {
			for (size_t col = 0; col < w; col++) {
				for (size_t ch = 0; ch < num_channels; ch++) {
					data[image_size*ch + row*w + col] = blob_image.at<Vec3b>(row, col)[ch];
				}
			}
		}
	}

	// 执行预测
	infer_request.Infer();

	// 处理预测结果 [1x1xNx7]
	for (auto &item : output_info) {
		auto output_name = item.first;

		// 获取输出数据
		auto output = infer_request.GetBlob(output_name);
		const float* detection = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output->buffer());
		const SizeVector outputDims = output->getTensorDesc().getDims();
		const int rows = outputDims[2];
		const int object_size = outputDims[3];

		// 解析输出结果
		for (int row = 0; row < rows; row++) {
			float label = detection[row*object_size + 1];
			float confidence = detection[row*object_size + 2];
			float x_min = detection[row*object_size + 3] * image_width;
			float y_min = detection[row*object_size + 4] * image_height;
			float x_max = detection[row*object_size + 5] * image_width;
			float y_max = detection[row*object_size + 6] * image_height;
			if (confidence > 0.5) {
				Rect object_box((int)x_min, (int)y_min, (int)(x_max - x_min), (int(y_max - y_min)));
				putText(src, "OpenVINO-2019R02 detection demo", Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2, 8);
				rectangle(src, object_box, Scalar(0, 0, 255), 2, 8, 0);
			}
		}
	}
	imshow("ssd-model optimizer demo", src);
	waitKey(0);
	destroyAllWindows();
	return 0;
}
