#include <CUDACommon.hpp>
#include <include/Classes.hpp>
#include <opencv2/cudaimgproc.hpp>

// cudaimgproc_EXPORTS is defined by CMake(add_library)
#if defined (_WIN32) && defined (_MSC_VER)
  #if defined(cudaimgproc_EXPORTS)
    #define  cudaimgproc_API __declspec(dllexport)
  #else
    #define  cudaimgproc_API __declspec(dllimport)
  #endif /* cudaimgproc_EXPORTS */
#else /* defined (_WIN32) */
 #define cudaimgproc_API
#endif

// CornernessCriteria

struct CornernessCriteriaPtr {
    void *ptr;
    inline cuda::CornernessCriteria * operator->() { return static_cast<cuda::CornernessCriteria *>(ptr); }
    inline CornernessCriteriaPtr(cuda::CornernessCriteria *ptr) { this->ptr = ptr; }
    inline cuda::CornernessCriteria & operator*() { return *static_cast<cuda::CornernessCriteria *>(this->ptr); }
};

// CornersDetector

struct CornersDetectorPtr {
    void *ptr;
    inline cuda::CornersDetector * operator->() { return static_cast<cuda::CornersDetector *>(ptr); }
    inline CornersDetectorPtr(cuda::CornersDetector *ptr) { this->ptr = ptr; }
    inline cuda::CornersDetector & operator*() { return *static_cast<cuda::CornersDetector *>(this->ptr); }
};

// TemplateMatching

struct TemplateMatchingPtr {
    void *ptr;
    inline cuda::TemplateMatching * operator->() { return static_cast<cuda::TemplateMatching *>(ptr); }
    inline TemplateMatchingPtr(cuda::TemplateMatching *ptr) { this->ptr = ptr; }
    inline cuda::TemplateMatching & operator*() { return *static_cast<cuda::TemplateMatching *>(this->ptr); }
};

extern "C" cudaimgproc_API
struct TensorWrapper cvtColorCuda(struct cutorchInfo info,
                                         struct TensorWrapper src, struct TensorWrapper dst, int code, int dstCn);

extern "C" cudaimgproc_API
struct TensorWrapper demosaicingCuda(struct cutorchInfo info,
                                 struct TensorWrapper src, struct TensorWrapper dst, int code, int dcn);

extern "C" cudaimgproc_API
void swapChannelsCuda(
        struct cutorchInfo info, struct TensorWrapper image,
        struct Vec4iWrapper dstOrder);

extern "C" cudaimgproc_API
struct TensorWrapper gammaCorrectionCuda(struct cutorchInfo info,
                                     struct TensorWrapper src, struct TensorWrapper dst, bool forward);

extern "C" cudaimgproc_API
struct TensorWrapper alphaCompCuda(struct cutorchInfo info,
                               struct TensorWrapper img1, struct TensorWrapper img2,
                               struct TensorWrapper dst, int alpha_op);

extern "C" cudaimgproc_API
struct TensorWrapper calcHistCuda(
        struct cutorchInfo info, struct TensorWrapper src, struct TensorWrapper hist);

extern "C" cudaimgproc_API
struct TensorWrapper equalizeHistCuda(struct cutorchInfo info,
                                  struct TensorWrapper src, struct TensorWrapper dst);

extern "C" cudaimgproc_API
struct TensorWrapper evenLevelsCuda(struct cutorchInfo info,
                                struct TensorWrapper levels, int nLevels, int lowerLevel, int upperLevel);

extern "C" cudaimgproc_API
struct TensorWrapper histEvenCuda(struct cutorchInfo info,
                              struct TensorWrapper src, struct TensorWrapper hist,
                              int histSize, int lowerLevel, int upperLevel);

extern "C" cudaimgproc_API
struct TensorArray histEven_4Cuda(struct cutorchInfo info,
                              struct TensorWrapper src, struct TensorArray hist, struct TensorWrapper histSize,
                              struct TensorWrapper lowerLevel, struct TensorWrapper upperLevel);

extern "C" cudaimgproc_API
struct TensorWrapper histRangeCuda(struct cutorchInfo info,
                               struct TensorWrapper src, struct TensorWrapper hist,
                               struct TensorWrapper levels);

extern "C" cudaimgproc_API
struct TensorArray histRange_4Cuda(struct cutorchInfo info,
                               struct TensorWrapper src, struct TensorArray hist, struct TensorWrapper levels);

extern "C" cudaimgproc_API
struct CornernessCriteriaPtr createHarrisCornerCuda(
        int srcType, int blockSize, int ksize, double k, int borderType);

extern "C" cudaimgproc_API
struct CornernessCriteriaPtr createMinEigenValCornerCuda(
        int srcType, int blockSize, int ksize, int borderType);

extern "C" cudaimgproc_API
struct TensorWrapper CornernessCriteria_computeCuda(
        struct cutorchInfo info, struct CornernessCriteriaPtr ptr,
        struct TensorWrapper src, struct TensorWrapper dst);

extern "C"
struct CornersDetectorPtr createGoodFeaturesToTrackDetectorCuda(
        int srcType, int maxCorners, double qualityLevel, double minDistance,
        int blockSize, bool useHarrisDetector, double harrisK);

extern "C" cudaimgproc_API
struct TensorWrapper CornersDetector_detectCuda(
        struct cutorchInfo info, struct CornersDetectorPtr ptr, struct TensorWrapper image,
        struct TensorWrapper corners, struct TensorWrapper mask);

extern "C" cudaimgproc_API
struct TemplateMatchingPtr createTemplateMatchingCuda(
        int srcType, int method, struct SizeWrapper user_block_size);

extern "C" cudaimgproc_API
struct TensorWrapper TemplateMatching_matchCuda(
        struct cutorchInfo info, struct TemplateMatchingPtr ptr, struct TensorWrapper image,
        struct TensorWrapper templ, struct TensorWrapper result);

extern "C" cudaimgproc_API
struct TensorWrapper bilateralFilterCuda(struct cutorchInfo info,
                                     struct TensorWrapper src, struct TensorWrapper dst, int kernel_size,
                                     float sigma_color, float sigma_spatial, int borderMode);

extern "C" cudaimgproc_API
struct TensorWrapper blendLinearCuda(struct cutorchInfo info,
                                 struct TensorWrapper img1, struct TensorWrapper img2, struct TensorWrapper weights1,
                                 struct TensorWrapper weights2, struct TensorWrapper result);
