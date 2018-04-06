#include <Common.hpp>
#include <opencv2/core.hpp>

// core_EXPORTS is defined by CMake(add_library)
#if defined (_WIN32) && defined (_MSC_VER)
  #if defined(core_EXPORTS)
    #define  core_API __declspec(dllexport)
  #else
    #define  core_API __declspec(dllimport)
  #endif /* core_EXPORTS */
#else /* defined (_WIN32) */
 #define core_API
#endif

extern "C" {

core_API int getNumThreads();

core_API void setNumThreads(int nthreads);

struct TensorWrapper copyMakeBorder(struct TensorWrapper src, struct TensorWrapper dst, int top, 
                                    int bottom, int left, int right, int borderType,
                                    struct ScalarWrapper value);
}