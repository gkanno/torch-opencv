#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/objdetect.hpp>

// objdetect_EXPORTS is defined by CMake(add_library)
#if defined (_WIN32) && defined (_MSC_VER)
  #if defined(objdetect_EXPORTS)
    #define  objdetect_API __declspec(dllexport)
  #else
    #define  objdetect_API __declspec(dllimport)
  #endif /* objdetect_EXPORTS */
#else /* defined (_WIN32) */
 #define objdetect_API
#endif

struct BaseCascadeClassifierPtr {
    void *ptr;

    inline cv::BaseCascadeClassifier * operator->() { return static_cast<cv::BaseCascadeClassifier *>(ptr); }
    inline BaseCascadeClassifierPtr(cv::BaseCascadeClassifier *ptr) { this->ptr = ptr; }
    inline cv::BaseCascadeClassifier & operator*() { return *static_cast<cv::BaseCascadeClassifier *>(this->ptr); }
};

struct CascadeClassifierPtr {
    void *ptr;

    inline cv::CascadeClassifier * operator->() { return static_cast<cv::CascadeClassifier *>(ptr); }
    inline CascadeClassifierPtr(cv::CascadeClassifier *ptr) { this->ptr = ptr; }
    inline cv::CascadeClassifier & operator*() { return *static_cast<cv::CascadeClassifier *>(this->ptr); }
};

struct HOGDescriptorPtr {
    void *ptr;

    inline cv::HOGDescriptor * operator->() { return static_cast<cv::HOGDescriptor *>(ptr); }
    inline HOGDescriptorPtr(cv::HOGDescriptor *ptr) { this->ptr = ptr; }
    inline cv::HOGDescriptor & operator*() { return *static_cast<cv::HOGDescriptor *>(this->ptr); }
};

extern "C" objdetect_API
struct TensorPlusRectArray groupRectangles(struct RectArray rectList, int groupThreshold, double eps);

extern "C" objdetect_API
bool BaseCascadeClassifier_empty(struct BaseCascadeClassifierPtr ptr);

extern "C" objdetect_API
bool BaseCascadeClassifier_load(struct BaseCascadeClassifierPtr ptr, const char *filename);

extern "C" objdetect_API
bool BaseCascadeClassifier_isOldFormatCascade(struct BaseCascadeClassifierPtr ptr);

extern "C" objdetect_API
struct SizeWrapper BaseCascadeClassifier_getOriginalWindowSize(struct BaseCascadeClassifierPtr ptr);

extern "C" objdetect_API
int BaseCascadeClassifier_getFeatureType(struct BaseCascadeClassifierPtr ptr);

extern "C" objdetect_API
struct CascadeClassifierPtr CascadeClassifier_ctor_default();

extern "C" objdetect_API
struct CascadeClassifierPtr CascadeClassifier_ctor(const char *filename);

extern "C" objdetect_API
void CascadeClassifier_dtor(struct CascadeClassifierPtr ptr);

extern "C" objdetect_API
bool CascadeClassifier_read(struct CascadeClassifierPtr ptr, struct FileNodePtr node);

extern "C" objdetect_API
struct RectArray CascadeClassifier_detectMultiScale(struct CascadeClassifierPtr ptr,
                                                    struct TensorWrapper image, double scaleFactor, int minNeighbors, int flags,
                                                    struct SizeWrapper minSize, struct SizeWrapper maxSize);

extern "C" objdetect_API
struct TensorPlusRectArray CascadeClassifier_detectMultiScale2(struct CascadeClassifierPtr ptr,
                                                               struct TensorWrapper image, double scaleFactor, int minNeighbors, int flags,
                                                               struct SizeWrapper minSize, struct SizeWrapper maxSize);

extern "C" objdetect_API
struct TensorArrayPlusRectArray CascadeClassifier_detectMultiScale3(
        struct CascadeClassifierPtr ptr, struct TensorWrapper image, double scaleFactor,
        int minNeighbors, int flags, struct SizeWrapper minSize, struct SizeWrapper maxSize,
        bool outputRejectLevels);

extern "C" objdetect_API
bool CascadeClassifier_convert(
        struct CascadeClassifierPtr ptr, const char *oldcascade, const char *newcascade);

extern "C" objdetect_API
struct HOGDescriptorPtr HOGDescriptor_ctor(
        struct SizeWrapper winSize, struct SizeWrapper blockSize, struct SizeWrapper blockStride,
        struct SizeWrapper cellSize, int nbins, int derivAperture, double winSigma,
        int histogramNormType, double L2HysThreshold, bool gammaCorrection,
        int nlevels, bool signedGradient);

extern "C" objdetect_API
void HOGDescriptor_dtor(struct HOGDescriptorPtr ptr);

extern "C" objdetect_API
size_t HOGDescriptor_getDescriptorSize(struct HOGDescriptorPtr ptr);

extern "C" objdetect_API
bool HOGDescriptor_checkDetectorSize(struct HOGDescriptorPtr ptr);

extern "C" objdetect_API
double HOGDescriptor_getWinSigma(struct HOGDescriptorPtr ptr);

extern "C" objdetect_API
void HOGDescriptor_setSVMDetector(struct HOGDescriptorPtr ptr, struct TensorWrapper _svmdetector);

extern "C" objdetect_API
bool HOGDescriptor_load(
        struct HOGDescriptorPtr ptr, const char *filename, const char *objname);

extern "C" objdetect_API
void HOGDescriptor_save(
        struct HOGDescriptorPtr ptr, const char *filename, const char *objname);

extern "C" objdetect_API
struct TensorWrapper HOGDescriptor_compute(
        struct HOGDescriptorPtr ptr, struct TensorWrapper img, struct SizeWrapper winStride,
        struct SizeWrapper padding, struct PointArray locations);

extern "C" objdetect_API
struct TensorPlusPointArray HOGDescriptor_detect(
        struct HOGDescriptorPtr ptr, struct TensorWrapper img, double hitThreshold,
        struct SizeWrapper winStride, struct SizeWrapper padding, struct PointArray searchLocations);

extern "C" objdetect_API
struct TensorPlusRectArray HOGDescriptor_detectMultiScale(
        struct HOGDescriptorPtr ptr, struct TensorWrapper img, double hitThreshold,
        struct SizeWrapper winStride, struct SizeWrapper padding, double scale,
        double finalThreshold, bool useMeanshiftGrouping);

extern "C" objdetect_API
struct TensorArray HOGDescriptor_computeGradient(
        struct HOGDescriptorPtr ptr, struct TensorWrapper img,
        struct SizeWrapper paddingTL, struct SizeWrapper paddingBR);

extern "C" objdetect_API
struct TensorWrapper HOGDescriptor_getDefaultPeopleDetector();

extern "C" objdetect_API
struct TensorWrapper HOGDescriptor_getDaimlerPeopleDetector();

