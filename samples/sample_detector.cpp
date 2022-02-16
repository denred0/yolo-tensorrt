#include "class_timer.hpp"
#include "class_detector.h"

#include <memory>
#include <thread>

#include <fstream>
#include <string>
#include <experimental/filesystem>
#include <filesystem>

#ifdef __linux__
	std::string separator = "/";
#elif _WIN32
	std::string separator = "\\";
#endif

std::vector<std::string> get_filenames(std::experimental::filesystem::path path)
{
	namespace stdfs = std::experimental::filesystem;

	std::vector<std::string> filenames;

	// http://en.cppreference.com/w/cpp/experimental/fs/directory_iterator
	const stdfs::directory_iterator end{};

	for (stdfs::directory_iterator iter{path}; iter != end; ++iter)
	{
		// http://en.cppreference.com/w/cpp/experimental/fs/is_regular_file
		if (stdfs::is_regular_file(*iter)) // comment out if all names (names of directories tc.) are required
			filenames.push_back(iter->path().string());
	}

	return filenames;
}

int main()
{
	tensor_rt::Config config_v3;
	config_v3.net_type = tensor_rt::YOLOV3;
	config_v3.file_model_cfg = "../configs/yolov3.cfg";
	config_v3.file_model_weights = "../configs/yolov3.weights";
	config_v3.calibration_image_list_file_txt = "../configs/calibration_images.txt";
	config_v3.inference_precison = tensor_rt::FP32;
	config_v3.detect_thresh = 0.5;

	tensor_rt::Config config_v4;
	config_v4.net_type = tensor_rt::YOLOV4;
	config_v4.file_model_cfg = "../samples/configs/yolov4/attributes/yolov4-obj-mycustom.cfg";
	config_v4.file_model_weights = "../samples/configs/yolov4/attributes/yolov4-obj-mycustom_best.weights";
	config_v4.calibration_image_list_file_txt = "../samples/configs/yolov4/attributes/calibration_images.txt";
	config_v4.inference_precison = tensor_rt::FP16;
	config_v4.detect_thresh = 0.3;
	config_v4.detect_nms = 0.3;
	config_v4.batch_size = 1;

	tensor_rt::Config config_v5;
	config_v5.net_type = tensor_rt::YOLOV5;
	config_v5.file_model_cfg = "../samples/configs/yolov5/attributes/256/yolov5m_attribites.cfg";
	config_v5.file_model_weights = "../samples/configs/yolov5/attributes/256/best.weights";
	config_v5.calibration_image_list_file_txt = "../samples/configs/yolov5/attributes/calibration_images.txt";
	config_v5.inference_precison = tensor_rt::FP16;
	config_v5.detect_thresh = 0.6;
	config_v5.detect_nms = 0.3;
	config_v5.batch_size = 1;

	std::unique_ptr<tensor_rt::Detector> detector(new tensor_rt::Detector());
	detector->init(config_v5);
	std::vector<tensor_rt::BatchResult> batch_res;

	std::string root_dir = "../samples/configs/yolov5/attributes/test_images";
	std::string output_dir = "../samples/configs/yolov5/attributes/test_images_result";
	std::vector<std::string> filenames = get_filenames(root_dir);
	int n_batch = 1; // config_v4.batch_size;

	Timer timer;
	cv::Mat frame;
	std::vector<cv::Mat> batch_img;
	int index = 0;
	int image_counter = 0;
	std::vector<std::string> bboxes_result;
	std::string result = "";

	using std::chrono::duration;
	using std::chrono::duration_cast;
	using std::chrono::high_resolution_clock;
	using std::chrono::milliseconds;

	double duration_time = 0;
	int inference_count = 0;

	auto t11 = high_resolution_clock::now();
	while (index < filenames.size())
	{

		// prepare batch data
		batch_img.clear();

		for (int bi = 0; bi < n_batch; ++bi)
		{
			frame = cv::imread(filenames[index]);
			index++;

			if (!frame.data)
				break;

			batch_img.push_back(frame);
		}

		bboxes_result.clear();

		// detect
		auto t1 = high_resolution_clock::now();
		timer.reset();
		detector->detect(batch_img, batch_res);
		timer.out("detect");
		auto t2 = high_resolution_clock::now();

		duration<double, std::milli> ms_double = t2 - t1;
		duration_time = duration_time + ms_double.count() / 1000;
		std::cout << "Inference time: " + std::to_string(ms_double.count() / 1000) + " sec \n";
		inference_count = inference_count + 1;

		// disp
		for (int i = 0; i < batch_img.size(); ++i)
		{
			for (const auto &r : batch_res[i])
			{
				std::cout << "batch " << i << " id:" << r.id << " prob:" << r.prob << " rect:" << r.rect << std::endl;
				cv::rectangle(batch_img[i], r.rect, cv::Scalar(255, 0, 0), 2);
				std::stringstream stream;
				stream << std::fixed << std::setprecision(2) << "id:" << r.id << "  score:" << r.prob;
				cv::putText(batch_img[i], stream.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5, cv::Scalar(0, 0, 255), 2);

				result = std::to_string(r.id) + " " +
						 std::to_string(r.prob) + " " +
						 std::to_string(r.rect.x) + " " +
						 std::to_string(r.rect.y) + " " +
						 std::to_string(r.rect.x + r.rect.width) + " " +
						 std::to_string(r.rect.y + r.rect.height);
				bboxes_result.push_back(result);
			}

			std::string filepath = filenames[image_counter];
			std::experimental::filesystem::path my_path{filepath};

			std::ofstream outFile(output_dir + separator + my_path.stem().string() + ".txt");
			for (const auto &e : bboxes_result)
				outFile << e << "\n";

			cv::imwrite(output_dir + separator + my_path.filename().string(), batch_img[i]);
			image_counter++;
		}
	}

	auto t22 = high_resolution_clock::now();

	duration<double, std::milli> ms_double = t22 - t11;

	std::cout << "FPS Inference only: " + std::to_string(filenames.size() / duration_time) + '\n';
	std::cout << "Duration: " + std::to_string(ms_double.count()) + '\n';
	std::cout << "Avg Inference time: " + std::to_string(duration_time / (filenames.size())) + '\n';
}
