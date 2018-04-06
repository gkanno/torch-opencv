#include <CUDACommon.hpp>
#include <include/Classes.hpp>
#include <opencv2/cudaarithm.hpp>

// cudaarithm_EXPORTS is defined by CMake(add_library)
#if defined (_WIN32) && defined (_MSC_VER)
  #if defined(cudaarithm_EXPORTS)
    #define  cudaarithm_API __declspec(dllexport)
  #else
    #define  cudaarithm_API __declspec(dllimport)
  #endif /* cudaarithm_EXPORTS */
#else /* defined (_WIN32) */
 #define cudaarithm_API
#endif

struct LookUpTablePtr {
    void *ptr;

    inline cuda::LookUpTable * operator->() { return static_cast<cuda::LookUpTable *>(ptr); }
    inline LookUpTablePtr(cuda::LookUpTable *ptr) { this->ptr = ptr; }
    inline operator const cuda::LookUpTable &() { return *static_cast<cuda::LookUpTable *>(ptr); }
};

struct ConvolutionPtr {
    void *ptr;

    inline cuda::Convolution * operator->() { return static_cast<cuda::Convolution *>(ptr); }
    inline ConvolutionPtr(cuda::Convolution *ptr) { this->ptr = ptr; }
    inline operator const cuda::Convolution &() { return *static_cast<cuda::Convolution *>(ptr); }
};

extern "C" cudaarithm_API
struct TensorWrapper minCuda(
        struct cutorchInfo info, struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper dst);

extern "C" cudaarithm_API
struct TensorWrapper maxCuda(
        struct cutorchInfo info, struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper dst);

extern "C" cudaarithm_API
struct TensorPlusDouble thresholdCuda(
        struct cutorchInfo info, struct TensorWrapper src,
        struct TensorWrapper dst, double thresh, double maxval, int type);

extern "C" cudaarithm_API
struct TensorWrapper magnitudeCuda(
        struct cutorchInfo info, struct TensorWrapper xy, struct TensorWrapper magnitude);

extern "C" cudaarithm_API
struct TensorWrapper magnitudeSqrCuda(
        struct cutorchInfo info, struct TensorWrapper xy, struct TensorWrapper magnitude);

extern "C" cudaarithm_API
struct TensorWrapper magnitude2Cuda(
        struct cutorchInfo info, struct TensorWrapper x, struct TensorWrapper y, struct TensorWrapper magnitude);

extern "C" cudaarithm_API
struct TensorWrapper magnitudeSqr2Cuda(
        struct cutorchInfo info, struct TensorWrapper x, struct TensorWrapper y, struct TensorWrapper magnitudeSqr);

extern "C" cudaarithm_API
struct TensorWrapper phaseCuda(
        struct cutorchInfo info, struct TensorWrapper x, struct TensorWrapper y,
        struct TensorWrapper angle, bool angleInDegrees);

extern "C" cudaarithm_API
struct TensorArray cartToPolarCuda(
        struct cutorchInfo info, struct TensorWrapper x, struct TensorWrapper y,
        struct TensorWrapper magnitude, struct TensorWrapper angle, bool angleInDegrees);

extern "C" cudaarithm_API
struct TensorArray polarToCartCuda(
        struct cutorchInfo info, struct TensorWrapper magnitude, struct TensorWrapper angle,
        struct TensorWrapper x, struct TensorWrapper y, bool angleInDegrees);

extern "C" cudaarithm_API
struct LookUpTablePtr LookUpTable_ctorCuda(
        struct cutorchInfo info, struct TensorWrapper lut);

extern "C" cudaarithm_API
struct TensorWrapper LookUpTable_transformCuda(
        struct cutorchInfo info, struct LookUpTablePtr ptr,
        struct TensorWrapper src, struct TensorWrapper dst);

extern "C" cudaarithm_API
struct TensorWrapper rectStdDevCuda(
        struct cutorchInfo info, struct TensorWrapper src, struct TensorWrapper sqr,
        struct TensorWrapper dst, struct RectWrapper rect);

extern "C" cudaarithm_API
struct TensorWrapper normalizeCuda(
        struct cutorchInfo info, struct TensorWrapper src, struct TensorWrapper dst,
        double alpha, double beta, int norm_type, int dtype, struct TensorWrapper mask);

extern "C" cudaarithm_API
struct TensorWrapper integralCuda(
        struct cutorchInfo info, struct TensorWrapper src, struct TensorWrapper sum);

extern "C" cudaarithm_API
struct TensorWrapper sqrIntegralCuda(
        struct cutorchInfo info, struct TensorWrapper src, struct TensorWrapper sum);

extern "C" cudaarithm_API
struct TensorWrapper mulSpectrumsCuda(
        struct cutorchInfo info, struct TensorWrapper src1, struct TensorWrapper src2,
        struct TensorWrapper dst, int flags, bool conjB);

extern "C" cudaarithm_API
struct TensorWrapper mulAndScaleSpectrumsCuda(
        struct cutorchInfo info, struct TensorWrapper src1, struct TensorWrapper src2,
        struct TensorWrapper dst, int flags, float scale, bool conjB);

extern "C" cudaarithm_API
struct TensorWrapper dftCuda(
        struct cutorchInfo info, struct TensorWrapper src,
        struct TensorWrapper dst, struct SizeWrapper dft_size, int flags);

extern "C" cudaarithm_API
struct ConvolutionPtr Convolution_ctorCuda(
        struct cutorchInfo info, struct SizeWrapper user_block_size);

extern "C" cudaarithm_API
struct TensorWrapper Convolution_convolveCuda(
        struct cutorchInfo info, struct ConvolutionPtr ptr, struct TensorWrapper image,
        struct TensorWrapper templ, struct TensorWrapper result, bool ccor);