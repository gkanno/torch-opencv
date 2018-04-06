#pragma once

extern "C" {
#include <TH/TH.h>
}

#include <opencv2/core.hpp>

#ifdef WITH_CUDA
#include <THC/THC.h>
#include <opencv2/core/cuda.hpp>

class GpuMatT;
#endif

#include <iostream>

// Common_EXPORTS is defined by CMake(add_library)
#if defined (_WIN32) && defined (_MSC_VER)
  #if defined(Common_EXPORTS)
    #define  Common_API __declspec(dllexport)
  #else
    #define  Common_API __declspec(dllimport)
  #endif /* Common_EXPORTS */
#else /* defined (_WIN32) */
 #define Common_API
#endif

extern "C" Common_API int getIntMax();
extern "C" Common_API float getFloatMax();
extern "C" Common_API double getDblEpsilon();

/***************** Tensor <=> Mat conversion *****************/

#define TO_MAT_OR_NOARRAY(mat) (mat.isNull() ? cv::noArray() : mat.toMat())
#define TO_GPUMAT_OR_NOARRAY(mat) (mat.isNull() ? cv::noArray() : mat.toGpuMat())

#define TO_MAT_LIST_OR_NOARRAY(mat) (mat.isNull() ? cv::noArray() : mat.toMatList())
#define TO_GPUMAT_LIST_OR_NOARRAY(mat) (mat.isNull() ? std::vector<cuda::GpuMat>() : mat.toGpuMatList())

extern "C" Common_API
void initAllocator();

class Common_API MatT {
public:
    cv::Mat mat;
    // The Tensor that `mat` was created from, or nullptr
    THByteTensor *tensor;

    operator cv::_InputOutputArray() { return this->mat; }

    MatT(cv::Mat && mat);
    MatT(cv::Mat & mat);
    MatT();
};

struct Common_API TensorWrapper {
    THByteTensor *tensorPtr;
    char typeCode;
    bool definedInLua;

    TensorWrapper();
    TensorWrapper(cv::Mat & mat);
    TensorWrapper(cv::Mat && mat);
    TensorWrapper(MatT & mat);
    TensorWrapper(MatT && mat);

    operator cv::Mat();
    // synonym for operator cv::Mat()
    cv::Mat toMat() { return *this; }
    MatT toMatT();

    #ifdef WITH_CUDA
    TensorWrapper(cv::cuda::GpuMat & mat, THCState *state);
    TensorWrapper(cv::cuda::GpuMat && mat, THCState *state);
    TensorWrapper(GpuMatT & mat, THCState *state);
    TensorWrapper(GpuMatT && mat, THCState *state);

    cv::cuda::GpuMat toGpuMat(int depth = -1);
    GpuMatT toGpuMatT();
    #endif

    bool isNull() { return tensorPtr == nullptr; }
};

struct Common_API TensorArray {
    struct TensorWrapper *tensors;
    int size;

    TensorArray();
    TensorArray(std::vector<cv::Mat> & matList);
    TensorArray(std::vector<MatT> & matList);
    explicit TensorArray(short size);

    #ifdef WITH_CUDA
    TensorArray(std::vector<cv::cuda::GpuMat> & matList, THCState *state);
    std::vector<cv::cuda::GpuMat> toGpuMatList();
    #endif

    operator std::vector<cv::Mat>();
    operator std::vector<MatT>();
    // synonyms for operators
    std::vector<cv::Mat> toMatList()  { return *this; }
    std::vector<MatT>    toMatTList() { return *this; }

    bool isNull() { return tensors == nullptr; }
};

inline
std::string typeStr(cv::Mat & mat) {
    switch (mat.depth()) {
        case CV_8U:  return "Byte";
        case CV_8S:  return "Char";
        case CV_16S: return "Short";
        case CV_32S: return "Int";
        case CV_32F: return "Float";
        case CV_64F: return "Double";
        default:     return "Unknown";
    }
}

extern "C" Common_API
void transfer_tensor(THByteTensor *dst, struct TensorWrapper srcWrapper);

	/***************** Wrappers for small OpenCV classes *****************/

struct Common_API SizeWrapper {
    int width, height;

    operator cv::Size() { return cv::Size(width, height); }
    SizeWrapper(const cv::Size & other);
    SizeWrapper() {}
};

struct Common_API Size2fWrapper {
    float width, height;

    operator cv::Size2f() { return cv::Size2f(width, height); }
    Size2fWrapper() {}
    Size2fWrapper(const cv::Size2f & other);
};

struct Common_API TermCriteriaWrapper {
    int type, maxCount;
    double epsilon;

    TermCriteriaWrapper() {}

    operator cv::TermCriteria() { return cv::TermCriteria(type, maxCount, epsilon); }
    TermCriteriaWrapper(cv::TermCriteria && other);
};

struct Common_API ScalarWrapper {
    double v0, v1, v2, v3;

    operator cv::Scalar() { return cv::Scalar(v0, v1, v2, v3); }
    ScalarWrapper(const cv::Scalar & other) {
        this->v0 = other.val[0];
        this->v1 = other.val[1];
        this->v2 = other.val[2];
        this->v3 = other.val[3];
    }
    ScalarWrapper() {}
};

struct Common_API Vec2dWrapper {
    double v0, v1;

    operator cv::Vec2d() { return cv::Vec2d(v0, v1); }
    Vec2dWrapper(const cv::Vec2d & other) {
        this->v0 = other.val[0];
        this->v1 = other.val[1];
    }
};

struct Common_API Vec3dWrapper {
    double v0, v1, v2;
    
    operator cv::Vec3d() { return cv::Vec3d(v0, v1, v2); }
    Vec3dWrapper & operator=(cv::Vec3d & other);
    Vec3dWrapper (const cv::Vec3d & other);
    Vec3dWrapper() {}
};

struct Common_API Vec3fWrapper {
    float v0, v1, v2;
};

struct Common_API Vec3iWrapper {
    int v0, v1, v2;
};

struct Common_API Vec4iWrapper {
    int v0, v1, v2, v3;
};

struct Common_API RectWrapper {
    int x, y, width, height;

    operator cv::Rect() { return cv::Rect(x, y, width, height); }
    RectWrapper & operator=(cv::Rect & other);
    RectWrapper(const cv::Rect & other);
    RectWrapper() {}
};

struct Common_API PointWrapper {
    int x, y;

    operator cv::Point() { return cv::Point(x, y); }

    PointWrapper() {}
    PointWrapper(const cv::Point & other);
};

struct Common_API Point2fWrapper {
    float x, y;

    operator cv::Point2f() { return cv::Point2f(x, y); }
    Point2fWrapper(const cv::Point2f & other);
    Point2fWrapper() {}
};

struct Common_API Point2dWrapper {
    double x, y;

    operator cv::Point2d() { return cv::Point2d(x, y); }
    Point2dWrapper(const cv::Point2d & other);
    Point2dWrapper() {}
};

struct Common_API RotatedRectWrapper {
    struct Point2fWrapper center;
    struct Size2fWrapper size;
    float angle;

    RotatedRectWrapper() {}
    RotatedRectWrapper(const cv::RotatedRect & other);
    operator cv::RotatedRect() { return cv::RotatedRect(center, size, angle); }
};

struct Common_API MomentsWrapper {
    double m00, m10, m01, m20, m11, m02, m30, m21, m12, m03;
    double mu20, mu11, mu02, mu30, mu21, mu12, mu03;
    double nu20, nu11, nu02, nu30, nu21, nu12, nu03;

    MomentsWrapper(const cv::Moments & other);
    operator cv::Moments() {
        return cv::Moments(m00, m10, m01, m20, m11, m02, m30, m21, m12, m03);
    }
};

struct Common_API RotatedRectPlusRect {
    struct RotatedRectWrapper rotrect;
    struct RectWrapper rect;
};

struct Common_API DMatchWrapper {
    int queryIdx;
    int trainIdx;
    int imgIdx;
    float distance;
};

struct Common_API DMatchArray {
    int size;
    struct DMatchWrapper *data;

    DMatchArray() {}
    DMatchArray(const std::vector<cv::DMatch> & other);
    operator std::vector<cv::DMatch>();
};

struct Common_API DMatchArrayOfArrays {
    int size;
    struct DMatchArray *data;

    DMatchArrayOfArrays() {}
    DMatchArrayOfArrays(const std::vector<std::vector<cv::DMatch>> & other);
    operator std::vector<std::vector<cv::DMatch>>();
};

struct Common_API KeyPointWrapper {
    struct Point2fWrapper pt;
    float size, angle, response;
    int octave, class_id;

    KeyPointWrapper(const cv::KeyPoint & other);
    operator cv::KeyPoint() { return cv::KeyPoint(pt, size, angle, response, octave, class_id); }
};

struct Common_API KeyPointArray {
    struct KeyPointWrapper *data;
    int size;

    KeyPointArray() {}
    KeyPointArray(const std::vector<cv::KeyPoint> & v);
    operator std::vector<cv::KeyPoint>();
};

/***************** Helper wrappers for [OpenCV class + some primitive] *****************/

struct Common_API TensorPlusDouble {
    struct TensorWrapper tensor;
    double val;
};

struct Common_API TensorPlusFloat {
    struct TensorWrapper tensor;
    float val;
};

struct Common_API TensorPlusInt {
    struct TensorWrapper tensor;
    int val;
};

struct Common_API TensorPlusBool {
    struct TensorWrapper tensor;
    bool val;
};

struct Common_API TensorPlusRect {
    struct TensorWrapper tensor;
    struct RectWrapper rect;
};

struct Common_API TensorPlusPoint {
    struct TensorWrapper tensor;
    struct PointWrapper point;
};

struct Common_API TensorArrayPlusFloat {
    struct TensorArray tensors;
    float val;
};

struct Common_API TensorArrayPlusDouble {
    struct TensorArray tensors;
    double val;
};

struct Common_API TensorArrayPlusInt {
    struct TensorArray tensors;
    int val;
};

struct Common_API TensorArrayPlusBool {
    struct TensorArray tensors;
    bool val;
};

struct Common_API TensorArrayPlusVec3d {
    struct TensorArray tensors;
    struct Vec3dWrapper vec3d;
};

struct Common_API TensorArrayPlusRect {
    struct TensorArray tensors;
    struct RectWrapper rect;
};

struct Common_API RectPlusInt {
    struct RectWrapper rect;
    int val;
};

struct Common_API RectPlusBool {
    struct RectWrapper rect;
    bool val;
};

struct Common_API ScalarPlusBool {
    struct ScalarWrapper scalar;
    bool val;
};

struct Common_API SizePlusInt {
    struct SizeWrapper size;
    int val;
};

struct Common_API Point2fPlusInt {
    struct Point2fWrapper point;
    int val;
};

/***************** Other helper structs *****************/

// Arrays

struct Common_API StringArray {
    char **data;
    int size;

    StringArray(int size):
        size(size),
        data(static_cast<char **>(malloc(size * sizeof(char*)))) {}

    operator std::vector<cv::String>();
};

struct Common_API UCharArray {
    unsigned char *data;
    int size;

    UCharArray() {}
    UCharArray(const std::vector<unsigned char> vec);
};

struct Common_API FloatArray {
    float *data;
    int size;

    FloatArray() {}
    FloatArray(const std::vector<float> vec);
};

struct Common_API DoubleArray {
    double *data;
    int size;

    DoubleArray() {}
    DoubleArray(const std::vector<double> vec);
};

struct Common_API PointArray {
    struct PointWrapper *data;
    int size;

    PointArray() {}
    PointArray(const std::vector<cv::Point> & vec);
    operator std::vector<cv::Point>();
};

struct Common_API RectArray {
    struct RectWrapper *data;
    int size;

    RectArray() {}
    RectArray(const std::vector<cv::Rect> & vec);
    operator std::vector<cv::Rect>();
};

struct Common_API SizeArray {
    struct SizeWrapper *data;
    int size;

    operator std::vector<cv::Size>();
};

struct Common_API TensorPlusRectArray {
    struct TensorWrapper tensor;
    struct RectArray rects;

    TensorPlusRectArray() {}
};

struct Common_API TensorArrayPlusRectArray {
    struct TensorArray tensors;
    struct RectArray rects;
};

struct Common_API TensorArrayPlusRectArrayPlusFloat {
    struct TensorArray tensors;
    struct RectArray rects;
    float val;
};

struct Common_API TensorPlusPointArray {
    struct TensorWrapper tensor;
    struct PointArray points;
};

struct Common_API TensorPlusKeyPointArray {
    struct TensorWrapper tensor;
    struct KeyPointArray keypoints;
};

// Arrays of arrays

struct Common_API FloatArrayOfArrays {
    float **pointers;
    float *realData;
    int dims;
};

struct Common_API PointArrayOfArrays {
    struct PointWrapper **pointers;
    struct PointWrapper *realData;
    int dims;
    int *sizes;
};

/***************** Helper functions *****************/

Common_API std::vector<MatT> get_vec_MatT(std::vector<cv::Mat> vec_mat);

Common_API std::vector<cv::UMat> get_vec_UMat(std::vector<cv::Mat> vec_mat);

Common_API std::vector<cv::Mat> get_vec_Mat(std::vector<cv::UMat> vec_umat);
