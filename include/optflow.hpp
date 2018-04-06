#include <Common.hpp>
#include <Classes.hpp>
#include <video.hpp>
#include <opencv2/optflow.hpp>

// optflow_EXPORTS is defined by CMake(add_library)
#if defined (_WIN32) && defined (_MSC_VER)
  #if defined(optflow_EXPORTS)
    #define  optflow_API __declspec(dllexport)
  #else
    #define  optflow_API __declspec(dllimport)
  #endif /* optflow_EXPORTS */
#else /* defined (_WIN32) */
 #define optflow_API
#endif

namespace optflow = cv::optflow;
namespace motempl = cv::motempl;

extern "C" optflow_API
struct TensorWrapper calcOpticalFlowSF(
        struct TensorWrapper from, struct TensorWrapper to, struct TensorWrapper flow,
        int layers, int averaging_block_size, int max_flow);

extern "C" optflow_API
struct TensorWrapper calcOpticalFlowSF_expanded(
        struct TensorWrapper from, struct TensorWrapper to, struct TensorWrapper flow,
        int layers, int averaging_block_size, int max_flow,
        double sigma_dist, double sigma_color, int postprocess_window,
        double sigma_dist_fix, double sigma_color_fix, double occ_thr,
        int upscale_averaging_radius, double upscale_sigma_dist,
        double upscale_sigma_color, double speed_up_thr);

extern "C" optflow_API
struct TensorWrapper readOpticalFlow(const char *path);

extern "C" optflow_API
bool writeOpticalFlow(const char *path, struct TensorWrapper flow);

extern "C" optflow_API
void updateMotionHistory(
        struct TensorWrapper silhouette, struct TensorWrapper mhi,
        double timestamp, double duration);

extern "C" optflow_API
struct TensorArray calcMotionGradient(
        struct TensorWrapper mhi, struct TensorWrapper mask, struct TensorWrapper orientation,
        double delta1, double delta2, int apertureSize);

extern "C" optflow_API
double calcGlobalOrientation(
        struct TensorWrapper orientation, struct TensorWrapper mask,
        struct TensorWrapper mhi, double timestamp, double duration);

extern "C" optflow_API
struct TensorPlusRectArray segmentMotion(
        struct TensorWrapper mhi, struct TensorWrapper segmask,
        double timestamp, double segThresh);

extern "C" optflow_API
struct DenseOpticalFlowPtr createOptFlow_DeepFlow_optflow();

extern "C" optflow_API
struct DenseOpticalFlowPtr createOptFlow_SimpleFlow_optflow();

extern "C" optflow_API
struct DenseOpticalFlowPtr createOptFlow_Farneback_optflow();

#if CV_MAJOR_VERSION >= 3 && CV_MINOR_VERSION >= 1

extern "C" optflow_API
struct TensorWrapper calcOpticalFlowSparseToDense(
        struct TensorWrapper from, struct TensorWrapper to, struct TensorWrapper flow,
        int grid_step, int k, float sigma, bool use_post_proc, float fgs_lambda, float fgs_sigma);

extern "C" optflow_API
struct DenseOpticalFlowPtr createOptFlow_SparseToDense_optflow();

#endif
