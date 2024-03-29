#include <sys/time.h>
#include <gflags/gflags.h>
#include <dirent.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <vector>
#include <fstream>
#include <sstream>

#include "../inc/utils.h"
#include "include/dataset/execute.h"
#include "include/dataset/transforms.h"
#include "include/dataset/vision.h"
#include "include/dataset/vision_ascend.h"
#include "include/api/types.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/api/context.h"

using mindspore::Serialization;
using mindspore::Model;
using mindspore::Context;
using mindspore::Status;
using mindspore::ModelType;
using mindspore::Graph;
using mindspore::GraphCell;
using mindspore::kSuccess;
using mindspore::MSTensor;
using mindspore::DataType;
using mindspore::dataset::Execute;
using mindspore::dataset::TensorTransform;
using mindspore::dataset::vision::Decode;
using mindspore::dataset::vision::Resize;
using mindspore::dataset::vision::Rescale;
using mindspore::dataset::vision::Normalize;
using mindspore::dataset::vision::HWC2CHW;
using mindspore::dataset::vision::HorizontalFlip;
using mindspore::dataset::vision::SwapRedBlue;
using mindspore::dataset::transforms::TypeCast;

DEFINE_string(model_path, "/PATH/TO/Auto-DeepLab-s.mindir", "model path");
DEFINE_string(dataset_path, "/PATH/TO/Cityscapes/leftImg8bit/val", "dataset path");
DEFINE_int32(device_id, 0, "device id");
DEFINE_string(precision_mode, "allow_fp32_to_fp16", "precision mode");
DEFINE_string(op_select_impl_mode, "", "op select impl mode");
DEFINE_string(device_target, "Ascend310", "device target");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (RealPath(FLAGS_model_path).empty()) {
        std::cout << "Invalid model" << std::endl;
        return 1;
    }

    auto context = std::make_shared<Context>();
    auto ascend310_info = std::make_shared<mindspore::Ascend310DeviceInfo>();
    ascend310_info->SetDeviceID(FLAGS_device_id);
    context->MutableDeviceInfo().push_back(ascend310_info);

    Graph graph;
    Status ret = Serialization::Load(FLAGS_model_path, ModelType::kMindIR, &graph);
    if (ret != kSuccess) {
        std::cout << "Load model failed." << std::endl;
        return 1;
    }

    Model model;
    ret = model.Build(GraphCell(graph), context);
    if (ret != kSuccess) {
        std::cout << "ERROR: Build failed." << std::endl;
        return 1;
    }

    std::vector<MSTensor> modelInputs = model.GetInputs();

    auto all_files = GetAllFiles(FLAGS_dataset_path);
    if (all_files.empty()) {
        std::cout << "ERROR: no input data." << std::endl;
        return 1;
    }

    auto decode = Decode();
    auto normalize = Normalize({123.675, 116.28, 103.53}, {58.395, 57.12, 57.375});
    auto hwc2chw = HWC2CHW();
    auto swapredblue = SwapRedBlue();
    auto flip = HorizontalFlip();
    auto typeCast = TypeCast(DataType::kNumberTypeFloat32);

    mindspore::dataset::Execute transformDecode({decode, swapredblue});
    mindspore::dataset::Execute transform({normalize, hwc2chw});
    mindspore::dataset::Execute transformFlip({normalize, flip, hwc2chw});
    mindspore::dataset::Execute transformCast(typeCast);

    std::map<double, double> costTime_map;

    size_t size = all_files.size();
    for (size_t i = 0; i < size; ++i) {
        struct timeval start;
        struct timeval end;
        double startTime_ms;
        double endTime_ms;
        std::vector<MSTensor> inputs;
        std::vector<MSTensor> flippedInputs;
        std::vector<MSTensor> outputs;
        std::vector<MSTensor> flippedOutputs;

        std::cout << "Start predict input files:" << all_files[i] << std::endl;
        mindspore::MSTensor image =  ReadFileToTensor(all_files[i]);
        mindspore::MSTensor flippedImage;

        ret = transformDecode(image, &image);
        if (ret != kSuccess) {
            std::cout << "ERROR: Decode failed." << std::endl;
            return 1;
        }
        std::vector<int64_t> shape = image.Shape();
        transformFlip(image, &flippedImage);
        transform(image, &image);

        inputs.emplace_back(modelInputs[0].Name(), modelInputs[0].DataType(), modelInputs[0].Shape(),
                            image.Data().get(), image.DataSize());
        flippedInputs.emplace_back(modelInputs[0].Name(), modelInputs[0].DataType(), modelInputs[0].Shape(),
                                   flippedImage.Data().get(), flippedImage.DataSize());

        gettimeofday(&start, NULL);
        model.Predict(inputs, &outputs);
        model.Predict(flippedInputs, &flippedOutputs);
        gettimeofday(&end, NULL);

        startTime_ms = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
        endTime_ms = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
        costTime_map.insert(std::pair<double, double>(startTime_ms, endTime_ms));
        std::string flippedName = all_files[i];
        flippedName.replace(flippedName.find('.'), flippedName.size() - flippedName.find('.'), "_flip.png");
        WriteResult(all_files[i], outputs);
        WriteResult(flippedName, flippedOutputs);
    }
    double average = 0.0;
    int infer_cnt = 0;

    for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
        double diff = 0.0;
        diff = iter->second - iter->first;
        average += diff;
        infer_cnt++;
    }

    average = average / infer_cnt;

    std::stringstream timeCost;
    timeCost << "NN inference cost average time: " << average << " ms of infer_count " << infer_cnt << std::endl;
    std::cout << "NN inference cost average time: " << average << "ms of infer_count " << infer_cnt << std::endl;
    std::string file_name = "./time_Result" + std::string("/test_perform_static.txt");
    std::ofstream file_stream(file_name.c_str(), std::ios::trunc);
    file_stream << timeCost.str();
    file_stream.close();
    costTime_map.clear();
    return 0;
}
