#include <Common.hpp>
#include <opencv2/imgcodecs.hpp>

// imgcodecs_EXPORTS is defined by CMake(add_library)
#if defined (_WIN32) && defined (_MSC_VER)
  #if defined(imgcodecs_EXPORTS)
    #define  imgcodecs_API __declspec(dllexport)
  #else
    #define  imgcodecs_API __declspec(dllimport)
  #endif /* imgcodecs_EXPORTS */
#else /* defined (_WIN32) */
 #define imgcodecs_API
#endif

extern "C" imgcodecs_API
struct TensorWrapper imread(const char *filename, int flags);

extern "C" imgcodecs_API
struct TensorArrayPlusBool imreadmulti(const char *filename, int flags);

extern "C" imgcodecs_API
bool imwrite(const char *filename, struct TensorWrapper img, struct TensorWrapper params);

extern "C" imgcodecs_API
struct TensorWrapper imdecode(struct TensorWrapper buf, int flags);

extern "C" imgcodecs_API
struct TensorWrapper imencode(
        const char *ext, struct TensorWrapper img, struct TensorWrapper params);