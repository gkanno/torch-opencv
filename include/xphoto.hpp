#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/xphoto.hpp>

// xphoto_EXPORTS is defined by CMake(add_library)
#if defined (_WIN32) && defined (_MSC_VER)
  #if defined(xphoto_EXPORTS)
    #define  xphoto_API __declspec(dllexport)
  #else
    #define  xphoto_API __declspec(dllimport)
  #endif /* xphoto_EXPORTS */
#else /* defined (_WIN32) */
 #define xphoto_API
#endif

extern "C" xphoto_API
struct TensorWrapper xphoto_autowbGrayworld(
        struct TensorWrapper src, struct TensorWrapper dst, float thresh);

extern "C" xphoto_API
struct TensorWrapper xphoto_balanceWhite(
        struct TensorWrapper src, struct TensorWrapper dst, int algorithmType,
        float inputMin, float inputMax, float outputMin, float outputMax);

extern "C" xphoto_API
struct TensorWrapper xphoto_dctDenoising(
        struct TensorWrapper src, struct TensorWrapper dst, double sigma, int psize);

extern "C" xphoto_API
struct TensorWrapper xphoto_inpaint(
        struct TensorWrapper src, struct TensorWrapper mask,
        struct TensorWrapper dst, int algorithmType);