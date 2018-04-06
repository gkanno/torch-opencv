#include <CUDACommon.hpp>
#include <include/Classes.hpp>
#include <opencv2/cudacodec.hpp>

// cudacodec_EXPORTS is defined by CMake(add_library)
#if defined (_WIN32) && defined (_MSC_VER)
  #if defined(cudacodec_EXPORTS)
    #define  cudacodec_API __declspec(dllexport)
  #else
    #define  cudacodec_API __declspec(dllimport)
  #endif /* cudacodec_EXPORTS */
#else /* defined (_WIN32) */
 #define cudacodec_API
#endif

namespace cudacodec = cv::cudacodec;

extern "C" cudacodec_API
struct cudacodec::EncoderParams EncoderParams_ctor_default();

extern "C" cudacodec_API
struct cudacodec::EncoderParams EncoderParams_ctor(const char *configFile);

extern "C" cudacodec_API
void EncoderParams_saveCuda(struct cudacodec::EncoderParams params, const char *configFile);

struct VideoWriterPtr {
    void *ptr;

    inline cudacodec::VideoWriter * operator->() { return static_cast<cudacodec::VideoWriter *>(ptr); }
    inline VideoWriterPtr(cudacodec::VideoWriter *ptr) { this->ptr = ptr; }
    inline cudacodec::VideoWriter & operator*() { return *static_cast<cudacodec::VideoWriter *>(this->ptr); }
};

struct VideoReaderPtr {
    void *ptr;

    inline cudacodec::VideoReader * operator->() { return static_cast<cudacodec::VideoReader *>(ptr); }
    inline VideoReaderPtr(cudacodec::VideoReader *ptr) { this->ptr = ptr; }
    inline cudacodec::VideoReader & operator*() { return *static_cast<cudacodec::VideoReader *>(this->ptr); }
};

extern "C" cudacodec_API
struct VideoWriterPtr VideoWriter_ctorCuda(
        const char *filename, struct SizeWrapper frameSize,
        double fps, struct cudacodec::EncoderParams params, int format);

extern "C" cudacodec_API
void VideoWriter_dtorCuda(struct VideoWriterPtr ptr);

extern "C" cudacodec_API
void VideoWriter_writeCuda(struct VideoWriterPtr ptr, struct TensorWrapper frame, bool lastFrame);

extern "C" cudacodec_API
struct cudacodec::EncoderParams VideoWriter_getEncoderParams(struct VideoWriterPtr ptr);

extern "C" cudacodec_API
struct VideoReaderPtr VideoReader_ctorCuda(const char *filename);

extern "C" cudacodec_API
void VideoReader_dtorCuda(struct VideoReaderPtr ptr);

extern "C" cudacodec_API
struct TensorWrapper VideoReader_nextFrameCuda(
        struct cutorchInfo info, struct VideoReaderPtr ptr, struct TensorWrapper frame);

extern "C" cudacodec_API
struct cudacodec::FormatInfo VideoReader_format(struct VideoReaderPtr ptr);