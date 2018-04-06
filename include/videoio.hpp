#include <Common.hpp>
#include <opencv2/videoio.hpp>

// videoio_EXPORTS is defined by CMake(add_library)
#if defined (_WIN32) && defined (_MSC_VER)
  #if defined(videoio_EXPORTS)
    #define  videoio_API __declspec(dllexport)
  #else
    #define  videoio_API __declspec(dllimport)
  #endif /* videoio_EXPORTS */
#else /* defined (_WIN32) */
 #define videoio_API
#endif

struct VideoCapturePtr {
    void *ptr;

    inline cv::VideoCapture * operator->() { return static_cast<cv::VideoCapture *>(ptr); }
    inline VideoCapturePtr(cv::VideoCapture *ptr) { this->ptr = ptr; }
    inline cv::VideoCapture & operator*() { return *static_cast<cv::VideoCapture *>(this->ptr); }
};

extern "C" videoio_API
struct VideoCapturePtr VideoCapture_ctor_default();

extern "C" videoio_API
struct VideoCapturePtr VideoCapture_ctor_device(int device);

extern "C" videoio_API
struct VideoCapturePtr VideoCapture_ctor_filename(const char *filename);

extern "C" videoio_API
void VideoCapture_dtor(VideoCapturePtr ptr);

extern "C" videoio_API
bool VideoCapture_open(VideoCapturePtr ptr, int device);

extern "C" videoio_API
bool VideoCapture_isOpened(VideoCapturePtr ptr);

extern "C" videoio_API
void VideoCapture_release(VideoCapturePtr ptr);

extern "C" videoio_API
bool VideoCapture_grab(VideoCapturePtr ptr);

extern "C" videoio_API
struct TensorPlusBool VideoCapture_retrieve(
        VideoCapturePtr ptr, struct TensorWrapper image, int flag);

extern "C" videoio_API
struct TensorPlusBool VideoCapture_read(
        VideoCapturePtr ptr, struct TensorWrapper image);

extern "C" videoio_API
bool VideoCapture_set(VideoCapturePtr ptr, int propId, double value);

extern "C" videoio_API
double VideoCapture_get(VideoCapturePtr ptr, int propId);

struct VideoWriterPtr {
    void *ptr;

    inline cv::VideoWriter * operator->() { return static_cast<cv::VideoWriter *>(ptr); }
    inline VideoWriterPtr(cv::VideoWriter *ptr) { this->ptr = ptr; }
    inline cv::VideoWriter & operator*() { return *static_cast<cv::VideoWriter *>(this->ptr); }
};

extern "C" videoio_API
struct VideoWriterPtr VideoWriter_ctor_default();

extern "C" videoio_API
struct VideoWriterPtr VideoWriter_ctor(
        const char *filename, int fourcc, double fps, struct SizeWrapper frameSize, bool isColor);

extern "C" videoio_API
void VideoWriter_dtor(struct VideoWriterPtr ptr);

extern "C" videoio_API
bool VideoWriter_open(struct VideoWriterPtr ptr, const char *filename, int fourcc,
                      double fps, struct SizeWrapper frameSize, bool isColor);

extern "C" videoio_API
bool VideoWriter_isOpened(struct VideoWriterPtr ptr);

extern "C" videoio_API
void VideoWriter_release(struct VideoWriterPtr ptr);

extern "C" videoio_API
void VideoWriter_write(struct VideoWriterPtr ptr, struct TensorWrapper image);

extern "C" videoio_API
bool VideoWriter_set(VideoWriterPtr ptr, int propId, double value);

extern "C" videoio_API
double VideoWriter_get(VideoWriterPtr ptr, int propId);

extern "C" videoio_API
int VideoWriter_fourcc(char c1, char c2, char c3, char c4);