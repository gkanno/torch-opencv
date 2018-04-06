#include <CUDACommon.hpp>
#include <include/Classes.hpp>
#include <opencv2/cudawarping.hpp>

// cudawarping_EXPORTS is defined by CMake(add_library)
#if defined (_WIN32) && defined (_MSC_VER)
  #if defined(cudawarping_EXPORTS)
    #define  cudawarping_API __declspec(dllexport)
  #else
    #define  cudawarping_API __declspec(dllimport)
  #endif /* cudawarping_EXPORTS */
#else /* defined (_WIN32) */
 #define cudawarping_API
#endif

extern "C" cudawarping_API
struct TensorWrapper remapCuda(struct cutorchInfo info,
                           struct TensorWrapper src, struct TensorWrapper map1,
                           struct TensorWrapper map2, int interpolation, struct TensorWrapper dst,
                           int borderMode, struct ScalarWrapper borderValue);

extern "C" cudawarping_API
struct TensorWrapper resizeCuda(struct cutorchInfo info,
                            struct TensorWrapper src, struct TensorWrapper dst,
                            struct SizeWrapper dsize, double fx, double fy,
                            int interpolation);

extern "C" cudawarping_API
struct TensorWrapper warpAffineCuda(struct cutorchInfo info,
                                struct TensorWrapper src, struct TensorWrapper dst,
                                struct TensorWrapper M, struct SizeWrapper dsize,
                                int flags, int borderMode, struct ScalarWrapper borderValue);

extern "C" cudawarping_API
struct TensorArray buildWarpAffineMapsCuda(
        struct cutorchInfo info, struct TensorWrapper M, bool inverse,
        struct SizeWrapper dsize, struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" cudawarping_API
struct TensorWrapper warpPerspectiveCuda(struct cutorchInfo info,
                                     struct TensorWrapper src, struct TensorWrapper dst,
                                     struct TensorWrapper M, struct SizeWrapper dsize,
                                     int flags, int borderMode, struct ScalarWrapper borderValue);

extern "C" cudawarping_API
struct TensorArray buildWarpPerspectiveMapsCuda(
        struct cutorchInfo info, struct TensorWrapper M, bool inverse,
        struct SizeWrapper dsize, struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" cudawarping_API
struct TensorWrapper rotateCuda(
        struct cutorchInfo info, struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dsize, double angle, double xShift, double yShift, int interpolation);

extern "C" cudawarping_API
struct TensorWrapper pyrDownCuda(struct cutorchInfo info,
                             struct TensorWrapper src, struct TensorWrapper dst);

extern "C" cudawarping_API
struct TensorWrapper pyrUpCuda(struct cutorchInfo info,
                           struct TensorWrapper src, struct TensorWrapper dst);
