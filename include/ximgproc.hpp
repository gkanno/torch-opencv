#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/ximgproc.hpp>

// ximgproc_EXPORTS is defined by CMake(add_library)
#if defined (_WIN32) && defined (_MSC_VER)
  #if defined(ximgproc_EXPORTS)
    #define  ximgproc_API __declspec(dllexport)
  #else
    #define  ximgproc_API __declspec(dllimport)
  #endif /* ximgproc_EXPORTS */
#else /* defined (_WIN32) */
 #define ximgproc_API
#endif

extern "C" ximgproc_API
struct TensorWrapper niBlackThreshold(struct TensorWrapper src, struct TensorWrapper dst, double maxValue, int type, int blockSize, double delta);


// GraphSegmentation
struct GraphSegmentationPtr {
    void *ptr;

    inline cv::ximgproc::segmentation::GraphSegmentation * operator->() { return static_cast<cv::ximgproc::segmentation::GraphSegmentation *>(ptr); }
    inline cv::ximgproc::segmentation::GraphSegmentation * operator*() { return static_cast<cv::ximgproc::segmentation::GraphSegmentation *>(ptr); }
    inline GraphSegmentationPtr(cv::ximgproc::segmentation::GraphSegmentation *ptr) { this->ptr = ptr; }
};

// GraphSegmentation
extern "C" ximgproc_API
struct GraphSegmentationPtr GraphSegmentation_ctor(double sigma, float k, int min_size);

extern "C" ximgproc_API
struct TensorWrapper GraphSegmentation_processImage(struct GraphSegmentationPtr ptr, struct TensorWrapper);

extern "C" ximgproc_API
void GraphSegmentation_setSigma(struct GraphSegmentationPtr ptr, double s);

extern "C" ximgproc_API
double GraphSegmentation_getSigma(struct GraphSegmentationPtr ptr);

extern "C" ximgproc_API
void GraphSegmentation_setK(struct GraphSegmentationPtr ptr, float k);

extern "C" ximgproc_API
float GraphSegmentation_getK(struct GraphSegmentationPtr ptr);

extern "C" ximgproc_API
void GraphSegmentation_setMinSize(struct GraphSegmentationPtr ptr, int min_size);

extern "C" ximgproc_API
int GraphSegmentation_getMinSize(struct GraphSegmentationPtr ptr);

// SuperpixelSLIC
struct SuperpixelSLICPtr {
    void *ptr;

    inline cv::ximgproc::SuperpixelSLIC * operator->() { return static_cast<cv::ximgproc::SuperpixelSLIC *>(ptr); }
    inline cv::ximgproc::SuperpixelSLIC * operator*() { return static_cast<cv::ximgproc::SuperpixelSLIC *>(ptr); }
    inline SuperpixelSLICPtr(cv::ximgproc::SuperpixelSLIC *ptr) { this->ptr = ptr; }
};

extern "C" ximgproc_API
struct SuperpixelSLICPtr SuperpixelSLIC_ctor(
        struct TensorWrapper image, int algorithm,
        int region_size, float ruler);

extern "C" ximgproc_API
int SuperpixelSLIC_getNumberOfSuperpixels(struct SuperpixelSLICPtr ptr);

extern "C" ximgproc_API
void SuperpixelSLIC_iterate(struct SuperpixelSLICPtr ptr, int num_iterations);

extern "C" ximgproc_API
struct TensorWrapper SuperpixelSLIC_getLabels(
        struct SuperpixelSLICPtr ptr, struct TensorWrapper labels_out);

extern "C" ximgproc_API
struct TensorWrapper SuperpixelSLIC_getLabelContourMask(
        struct SuperpixelSLICPtr ptr, struct TensorWrapper image, bool thick_line);

extern "C" ximgproc_API
void SuperpixelSLIC_enforceLabelConnectivity(
        struct SuperpixelSLICPtr ptr, int min_element_size);

// See #103 and #95
/*

// SelectiveSearchSegmentationStrategy
struct SelectiveSearchSegmentationStrategyPtr {
    void *ptr;

    inline cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategy * operator->() { return static_cast<cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategy *>(ptr); }
    inline cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategy * operator*() { return static_cast<cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategy *>(ptr); }
    inline SelectiveSearchSegmentationStrategyPtr(cv::ximgproc::segmentation::SelectiveSearchSegmentationStrategy *ptr) { this->ptr = ptr; }
};

//
// extern "C" ximgproc_API
// struct SelectiveSearchSegmentationStrategyPtr SelectiveSearchSegmentationStrategyColor_ctor();
//
// extern "C" ximgproc_API
// struct SelectiveSearchSegmentationStrategyPtr SelectiveSearchSegmentationStrategySize_ctor();
//
// extern "C" ximgproc_API
// struct SelectiveSearchSegmentationStrategyPtr SelectiveSearchSegmentationStrategyTexture_ctor();
//
// extern "C" ximgproc_API
// struct SelectiveSearchSegmentationStrategyPtr SelectiveSearchSegmentationStrategyFill_ctor();
//
// extern "C" ximgproc_API
// void SelectiveSearchSegmentationStrategy_setImage(struct SelectiveSearchSegmentationStrategyPtr ptr, struct TensorWrapper, struct TensorWrapper, struct TensorWrapper, int);
//
// extern "C" ximgproc_API
// float SelectiveSearchSegmentationStrategy_get(int, int);
//
// extern "C" ximgproc_API
// void SelectiveSearchSegmentationStrategy_merge(int, int);


// MULTIPLE STRTEGY
//

// SelectiveSearchSegmentation
struct SelectiveSearchSegmentationPtr {
    void *ptr;

    inline cv::ximgproc::segmentation::SelectiveSearchSegmentation * operator->() { return static_cast<cv::ximgproc::segmentation::SelectiveSearchSegmentation *>(ptr); }
    inline cv::ximgproc::segmentation::SelectiveSearchSegmentation * operator*() { return static_cast<cv::ximgproc::segmentation::SelectiveSearchSegmentation *>(ptr); }
    inline SelectiveSearchSegmentationPtr(cv::ximgproc::segmentation::SelectiveSearchSegmentation *ptr) { this->ptr = ptr; }
};

extern "C" ximgproc_API
struct SelectiveSearchSegmentationPtr SelectiveSearchSegmentation_ctor();

extern "C" ximgproc_API
void SelectiveSearchSegmentation_setBaseImage(struct SelectiveSearchSegmentationPtr ptr, struct TensorWrapper);

extern "C" ximgproc_API
void SelectiveSearchSegmentation_switchToSingleStrategy(struct SelectiveSearchSegmentationPtr ptr, int, float);

extern "C" ximgproc_API
void SelectiveSearchSegmentation_switchToSelectiveSearchFast(struct SelectiveSearchSegmentationPtr ptr, int, int, float);

extern "C" ximgproc_API
void SelectiveSearchSegmentation_switchToSelectiveSearchQuality(struct SelectiveSearchSegmentationPtr ptr, int, int, float);

extern "C" ximgproc_API
void SelectiveSearchSegmentation_addImage(struct SelectiveSearchSegmentationPtr ptr, struct TensorWrapper);

extern "C" ximgproc_API
void SelectiveSearchSegmentation_clearImages(struct SelectiveSearchSegmentationPtr ptr);

extern "C" ximgproc_API
void SelectiveSearchSegmentation_addGraphSegmentation(struct SelectiveSearchSegmentationPtr ptr, struct GraphSegmentationPtr);

extern "C" ximgproc_API
void SelectiveSearchSegmentation_clearGraphSegmentations(struct SelectiveSearchSegmentationPtr ptr);

extern "C" ximgproc_API
void SelectiveSearchSegmentation_addStrategy(struct SelectiveSearchSegmentationPtr ptr, struct SelectiveSearchSegmentationStrategyPtr);

extern "C" ximgproc_API
void SelectiveSearchSegmentation_clearStrategies(struct SelectiveSearchSegmentationPtr ptr);

extern "C" ximgproc_API
struct RectArray SelectiveSearchSegmentation_process(struct SelectiveSearchSegmentationPtr ptr);
*/