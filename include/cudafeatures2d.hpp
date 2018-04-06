#include <CUDACommon.hpp>
#include <include/Classes.hpp>
#include <opencv2/cudafeatures2d.hpp>

// cudafeatures2d_EXPORTS is defined by CMake(add_library)
#if defined (_WIN32) && defined (_MSC_VER)
  #if defined(cudafeatures2d_EXPORTS)
    #define  cudafeatures2d_API __declspec(dllexport)
  #else
    #define  cudafeatures2d_API __declspec(dllimport)
  #endif /* cudafeatures2d_EXPORTS */
#else /* defined (_WIN32) */
 #define cudafeatures2d_API
#endif

// DescriptorMatcher

struct DescriptorMatcherPtr {
    void *ptr;
    inline cuda::DescriptorMatcher * operator->() { return static_cast<cuda::DescriptorMatcher *>(ptr); }
    inline DescriptorMatcherPtr(cuda::DescriptorMatcher *ptr) { this->ptr = ptr; }
    inline cuda::DescriptorMatcher & operator*() { return *static_cast<cuda::DescriptorMatcher *>(this->ptr); }
};

// Feature2DAsync

struct Feature2DAsyncPtr {
    void *ptr;
    inline cuda::Feature2DAsync * operator->() { return static_cast<cuda::Feature2DAsync *>(ptr); }
    inline Feature2DAsyncPtr(cuda::Feature2DAsync *ptr) { this->ptr = ptr; }
    inline cuda::Feature2DAsync & operator*() { return *static_cast<cuda::Feature2DAsync *>(this->ptr); }
};

// FastFeatureDetector

struct FastFeatureDetectorPtr {
    void *ptr;
    inline cuda::FastFeatureDetector * operator->() { return static_cast<cuda::FastFeatureDetector *>(ptr); }
    inline FastFeatureDetectorPtr(cuda::FastFeatureDetector *ptr) { this->ptr = ptr; }
    inline cuda::FastFeatureDetector & operator*() { return *static_cast<cuda::FastFeatureDetector *>(this->ptr); }
};

// ORB

struct ORBPtr {
    void *ptr;
    inline cuda::ORB * operator->() { return static_cast<cuda::ORB *>(ptr); }
    inline ORBPtr(cuda::ORB *ptr) { this->ptr = ptr; }
    inline cuda::ORB & operator*() { return *static_cast<cuda::ORB *>(this->ptr); }
};

extern "C" cudafeatures2d_API
struct DescriptorMatcherPtr createBFMatcherCuda(int normType);

extern "C" cudafeatures2d_API
bool DescriptorMatcher_isMaskSupportedCuda(struct DescriptorMatcherPtr ptr);

extern "C" cudafeatures2d_API
void DescriptorMatcher_addCuda(
        struct DescriptorMatcherPtr ptr, struct TensorArray descriptors);

extern "C" cudafeatures2d_API
struct TensorArray DescriptorMatcher_getTrainDescriptorsCuda(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr);

extern "C" cudafeatures2d_API
void DescriptorMatcher_clearCuda(struct DescriptorMatcherPtr ptr);

extern "C" cudafeatures2d_API
bool DescriptorMatcher_emptyCuda(struct DescriptorMatcherPtr ptr);

extern "C" cudafeatures2d_API
void DescriptorMatcher_trainCuda(struct DescriptorMatcherPtr ptr);

extern "C" cudafeatures2d_API
struct TensorWrapper DescriptorMatcher_matchCuda(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, struct TensorWrapper mask);

extern "C" cudafeatures2d_API
struct TensorWrapper DescriptorMatcher_match_masksCuda(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper matches,
        struct TensorArray masks);

extern "C" cudafeatures2d_API
struct DMatchArray DescriptorMatcher_matchConvertCuda(
        struct DescriptorMatcherPtr ptr, struct TensorWrapper gpu_matches);

extern "C" cudafeatures2d_API
struct TensorWrapper DescriptorMatcher_knnMatchCuda(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, int k, struct TensorWrapper mask);

extern "C" cudafeatures2d_API
struct TensorWrapper DescriptorMatcher_knnMatch_masksCuda(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, int k, struct TensorArray masks);

extern "C" cudafeatures2d_API
struct DMatchArrayOfArrays DescriptorMatcher_knnMatchConvertCuda(
        struct DescriptorMatcherPtr ptr,
        struct TensorWrapper gpu_matches, bool compactResult);

extern "C" cudafeatures2d_API
struct TensorWrapper DescriptorMatcher_radiusMatchCuda(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, float maxDistance, struct TensorWrapper mask);

extern "C" cudafeatures2d_API
struct TensorWrapper DescriptorMatcher_radiusMatch_masksCuda(
        struct cutorchInfo info, struct DescriptorMatcherPtr ptr,
        struct TensorWrapper queryDescriptors, struct TensorWrapper trainDescriptors,
        struct TensorWrapper matches, float maxDistance, struct TensorArray masks);

extern "C" cudafeatures2d_API
struct DMatchArrayOfArrays DescriptorMatcher_radiusMatchConvertCuda(
        struct DescriptorMatcherPtr ptr,
        struct TensorWrapper gpu_matches, bool compactResult);

extern "C" cudafeatures2d_API
void Feature2DAsync_dtorCuda(struct Feature2DAsyncPtr ptr);

extern "C" cudafeatures2d_API
struct TensorWrapper Feature2DAsync_detectAsyncCuda(
        struct cutorchInfo info, struct Feature2DAsyncPtr ptr, struct TensorWrapper image,
        struct TensorWrapper keypoints, struct TensorWrapper mask);

extern "C" cudafeatures2d_API
struct TensorArray Feature2DAsync_computeAsyncCuda(
        struct cutorchInfo info, struct Feature2DAsyncPtr ptr, struct TensorWrapper image,
        struct TensorWrapper keypoints, struct TensorWrapper descriptors);

extern "C" cudafeatures2d_API
struct TensorArray Feature2DAsync_detectAndComputeAsyncCuda(
        struct cutorchInfo info, struct Feature2DAsyncPtr ptr, struct TensorWrapper image,
        struct TensorWrapper mask, struct TensorWrapper keypoints,
        struct TensorWrapper descriptors, bool useProvidedKeypoints);

extern "C" cudafeatures2d_API
struct KeyPointArray Feature2DAsync_convertCuda(
        struct Feature2DAsyncPtr ptr, struct TensorWrapper gpu_keypoints);

extern "C" cudafeatures2d_API
struct FastFeatureDetectorPtr FastFeatureDetector_ctorCuda(
        int threshold, bool nonmaxSuppression, int type, int max_npoints);

extern "C" cudafeatures2d_API
void FastFeatureDetector_dtorCuda(struct FastFeatureDetectorPtr ptr);

extern "C" cudafeatures2d_API
void FastFeatureDetector_setMaxNumPointsCuda(struct FastFeatureDetectorPtr ptr, int val);

extern "C" cudafeatures2d_API
int FastFeatureDetector_getMaxNumPointsCuda(struct FastFeatureDetectorPtr ptr);

extern "C" cudafeatures2d_API
struct ORBPtr ORB_ctorCuda(
        int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, int firstLevel,
        int WTA_K, int scoreType, int patchSize, int fastThreshold, bool blurForDescriptor);

extern "C" cudafeatures2d_API
void ORB_setBlurForDescriptorCuda(struct ORBPtr ptr, bool val);

extern "C" cudafeatures2d_API
bool ORB_getBlurForDescriptorCuda(struct ORBPtr ptr);
