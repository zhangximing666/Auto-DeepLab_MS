#ifndef AUTODEEPLAB_H
#define AUTODEEPLAB_H

#include <sys/stat.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>

#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "SegmentPostProcessors/Deeplabv3Post.h"
#include "MxBase/PostProcessBases/PostProcessDataType.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    std::string modelPath;
    uint32_t classNum;
    uint32_t modelType;
    bool checkModel;
    uint32_t frameworkType;
};

class AutoDeepLab {
public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR ReadImage(const std::string &imgPath, cv::Mat &imageMat);
    APP_ERROR Normalize(const cv::Mat &srcImageMat, cv::Mat &dstImageMat);
    APP_ERROR GetResizeInfo(const cv::Mat &srcImageMat, MxBase::ResizedImageInfo &resizedImageInfo);
    APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase);
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &inputs,
        std::vector<MxBase::SemanticSegInfo> &segInfo, const std::vector<MxBase::ResizedImageInfo> &resizedInfo);
    APP_ERROR Process(const std::string &imgPath);
    APP_ERROR SaveResultToImage(const MxBase::SemanticSegInfo &segInfo, const std::string &filePath);

private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
    std::shared_ptr<MxBase::Deeplabv3Post> post_;
    MxBase::ModelDesc modelDesc_;
    uint32_t deviceId_ = 0;
};

APP_ERROR GetAllImages(const std::string &dirName, std::vector<std::string> *ImagesPath);
DIR *OpenDir(const std::string &dirName);
std::string RealPath(const std::string &path);
#endif
