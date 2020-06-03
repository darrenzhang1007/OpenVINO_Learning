#include <inference_engine.hpp>
#include <ext_list.hpp>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace InferenceEngine;
using namespace std;
static std::vector<std::string> items = {
	"neutral", "happy", "sad", "surprise", "anger" };


template <typename T>
void matU8ToBlob(const cv::Mat& orig_image, InferenceEngine::Blob::Ptr& blob, int batchIndex = 0) {
	InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
	const size_t width = blobSize[3];
	const size_t height = blobSize[2];
	const size_t channels = blobSize[1];
	T* blob_data = blob->buffer().as<T*>();

	cv::Mat resized_image(orig_image);
	if (width != orig_image.size().width || height != orig_image.size().height) {
		cv::resize(orig_image, resized_image, cv::Size(width, height));
	}

	// 耗时操作！！
	for (size_t h = 0; h < height; h++) {
		uchar* curr_row = resized_image.ptr<uchar>(h);
		for (size_t w = 0; w < width; w++) {
			for (size_t c = 0; c < channels; c++) {
				blob_data[c * width * height + h * width + w] = *curr_row++;
			}
		}
	}
}


void frametoBlob(const Mat &frame, InferRequest::Ptr &inferRequest, const std::string & inputName) {
	Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
	matU8ToBlob<uint8_t>(frame, frameBlob);
}


int main(int argc, char** argv) {
	string bin = "E:/models/tf_models/face-detection-adas-0001/face-detection-adas-0001.bin";
	string xml = "E:/models/tf_models/face-detection-adas-0001/face-detection-adas-0001.xml";
	namedWindow("frame", WINDOW_AUTOSIZE);
	namedWindow("face detection", WINDOW_AUTOSIZE);

	typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

	vector<file_name_t> dirs;
	std::string s("C:/Program Files (x86)/IntelSWTools/openvino_2019.2.242/deployment_tools/inference_engine/bin/intel64/Debug");
	std::wstring ws;
	ws.assign(s.begin(), s.end());
	dirs.push_back(ws);

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

	CNNNetReader emotion_reader;
	emotion_reader.ReadNetwork("E:/models/tf_models/emotions-recognition-retail-0003/emotions-recognition-retail-0003.xml");
	emotion_reader.ReadWeights("E:/models/tf_models/emotions-recognition-retail-0003/emotions-recognition-retail-0003.bin");

	// 获取输入输出
	auto network = network_reader.getNetwork();
	InferenceEngine::InputsDataMap input_info(network.getInputsInfo());
	InferenceEngine::OutputsDataMap output_info(network.getOutputsInfo());

	auto emotion_network = emotion_reader.getNetwork();
	InferenceEngine::InputsDataMap em_input_info(emotion_network.getInputsInfo());
	InferenceEngine::OutputsDataMap em_output_info(emotion_network.getOutputsInfo());
	
	// 设置输入
	for (auto &item : input_info) {
		auto input_data = item.second;
		input_data->setPrecision(Precision::U8);
		input_data->setLayout(Layout::NCHW);
	}
	auto inputName = input_info.begin()->first;
	InputInfo::Ptr& em_input = em_input_info.begin()->second;
	auto em_inputName = em_input_info.begin()->first;
	em_input->setPrecision(Precision::U8);
	em_input->getInputData()->setLayout(Layout::NCHW);

	// 设置输出
	for (auto &item : output_info) {
		auto output_data = item.second;
		output_data->setPrecision(Precision::FP32);
	}
	DataPtr em_output = em_output_info.begin()->second;
	auto em_outputName = em_output_info.begin()->first;
	const SizeVector outputDims = em_output->getTensorDesc().getDims();
	const int em_maxProposalCount = outputDims[2];
	const int me_objectSize = outputDims[3];
	em_output->setPrecision(Precision::FP32);
	em_output->setLayout(Layout::NCHW);

	// 创建可执行网络
	auto executable_network = ie.LoadNetwork(network, "CPU");
	auto em_exec_network = ie.LoadNetwork(emotion_network, "CPU");

	// 请求推断
	auto infer_request_curr = executable_network.CreateInferRequestPtr();
	auto infer_request_next = executable_network.CreateInferRequestPtr();
	auto em_infer = em_exec_network.CreateInferRequestPtr();

	// 开视频窗口
	VideoCapture capture("H:/OpenVINO_Learning/006_face-detection/test.mp4 ");

	Mat curr_frame, next_frame;
	capture.read(curr_frame);
	int image_width = curr_frame.cols;
	int image_height = curr_frame.rows;
	bool isLastFrame = false;
	bool isFirstFrame = true;

	frametoBlob(curr_frame, infer_request_curr, inputName);
	while (true) {
		// 读取每一帧
		if (!capture.read(next_frame)) {
			if (next_frame.empty()) {
				isLastFrame = true;
			}
		}
		auto t0 = std::chrono::high_resolution_clock::now();

		if (!isLastFrame) {
			frametoBlob(next_frame, infer_request_next, inputName);
		}
		// 开启异步执行模式
		if (isFirstFrame) {
			infer_request_curr->StartAsync();
			infer_request_next->StartAsync();
			isFirstFrame = false;
		}
		else {
			if (!isLastFrame) {
				infer_request_next->StartAsync();
			}
		}
		imshow("frame", curr_frame);

		// 检查返回数据
		if (OK == infer_request_curr->Wait(IInferRequest::WaitMode::RESULT_READY)) {
			auto output_name = output_info.begin()->first;
			auto output = infer_request_curr->GetBlob(output_name);

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
				if (confidence > 0.7) {
					Rect object_box((int)x_min, (int)y_min, (int)(x_max - x_min), (int(y_max - y_min)));
					if (object_box.x >= 0 && object_box.y >= 0 &&
						(object_box.x + object_box.width) < curr_frame.cols &&
						(object_box.y + object_box.height) < curr_frame.rows) {
						Mat face = curr_frame(object_box);
						frametoBlob(face, em_infer, em_inputName);
						em_infer->Infer();

						// 解析结果
						auto em_data = em_infer->GetBlob(em_outputName)->buffer().as<float*>();
						const auto em_id = max_element(em_data, em_data + 5) - em_data;
						putText(curr_frame, items[em_id].c_str(), Point(x_min, y_min + 20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2, 8);
					}
					rectangle(curr_frame, object_box, Scalar(0, 0, 255), 2, 8, 0);
				}
			}
			// 计算执行时间
			auto t1 = std::chrono::high_resolution_clock::now();
			ms dtime = std::chrono::duration_cast<ms>(t1 - t0);
			ostringstream ss;
			ss << "detection time : " << std::fixed << std::setprecision(2) << dtime.count() << "ms"
				<< "(" << (1000 / dtime.count()) << "fps)";
			putText(curr_frame, ss.str(), Point(10, 35), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);
		}
		imshow("face detection", curr_frame);
		char c = waitKey(1);
		if (c == 27) {
			break;
		}
		if (isLastFrame) {
			break;
		}

		// 异步交换
		next_frame.copyTo(curr_frame);
		infer_request_curr.swap(infer_request_next);
	}

	waitKey(0);
	destroyAllWindows();
	return 0;
}
