#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/superres.hpp>

// superres_EXPORTS is defined by CMake(add_library)
#if defined (_WIN32) && defined (_MSC_VER)
  #if defined(superres_EXPORTS)
    #define  superres_API __declspec(dllexport)
  #else
    #define  superres_API __declspec(dllimport)
  #endif /* superres_EXPORTS */
#else /* defined (_WIN32) */
 #define superres_API
#endif

namespace superres = cv::superres;

struct FrameSourcePtr {
    void *ptr;

    inline superres::FrameSource * operator->() { return static_cast<superres::FrameSource *>(ptr); }
    inline FrameSourcePtr(superres::FrameSource *ptr) { this->ptr = ptr; }
};

extern "C" superres_API
struct FrameSourcePtr createFrameSource();

extern "C" superres_API
struct FrameSourcePtr createFrameSource_Video(const char *fileName);

extern "C" superres_API
struct FrameSourcePtr createFrameSource_Video_CUDA(const char *fileName);

extern "C" superres_API
struct FrameSourcePtr createFrameSource_Camera(int deviceId);

extern "C" superres_API
void FrameSource_dtor(struct FrameSourcePtr ptr);

extern "C" superres_API
struct TensorWrapper FrameSource_nextFrame(struct FrameSourcePtr ptr, struct TensorWrapper frame);

extern "C" superres_API
void FrameSource_reset(struct FrameSourcePtr ptr);

struct SuperResolutionPtr {
    void *ptr;

    inline superres::SuperResolution * operator->() { return static_cast<superres::SuperResolution *>(ptr); }
    inline SuperResolutionPtr(superres::SuperResolution *ptr) { this->ptr = ptr; }
};

extern "C" superres_API
struct SuperResolutionPtr createSuperResolution_BTVL1();

extern "C" superres_API
struct SuperResolutionPtr createSuperResolution_BTVL1_CUDA();

extern "C" superres_API
struct TensorWrapper SuperResolution_nextFrame(struct SuperResolutionPtr ptr, struct TensorWrapper frame);

extern "C" superres_API
void SuperResolution_reset(struct SuperResolutionPtr ptr);

extern "C" superres_API
void SuperResolution_setInput(struct SuperResolutionPtr ptr, struct FrameSourcePtr frameSource);

extern "C" superres_API
void SuperResolution_collectGarbage(struct SuperResolutionPtr ptr);

extern "C" superres_API
void SuperResolution_setScale(struct SuperResolutionPtr ptr, int val);

extern "C" superres_API
int SuperResolution_getScale(struct SuperResolutionPtr ptr);

extern "C" superres_API
void SuperResolution_setIterations(struct SuperResolutionPtr ptr, int val);

extern "C" superres_API
int SuperResolution_getIterations(struct SuperResolutionPtr ptr);

extern "C" superres_API
void SuperResolution_setTau(struct SuperResolutionPtr ptr, double val);

extern "C" superres_API
double SuperResolution_getTau(struct SuperResolutionPtr ptr);

extern "C" superres_API
void SuperResolution_setLabmda(struct SuperResolutionPtr ptr, double val);

extern "C" superres_API
double SuperResolution_getLabmda(struct SuperResolutionPtr ptr);

extern "C" superres_API
void SuperResolution_setAlpha(struct SuperResolutionPtr ptr, double val);

extern "C" superres_API
double SuperResolution_getAlpha(struct SuperResolutionPtr ptr);

extern "C" superres_API
void SuperResolution_setKernelSize(struct SuperResolutionPtr ptr, int val);

extern "C" superres_API
int SuperResolution_getKernelSize(struct SuperResolutionPtr ptr);

extern "C" superres_API
void SuperResolution_setBlurKernelSize(struct SuperResolutionPtr ptr, int val);

extern "C" superres_API
int SuperResolution_getBlurKernelSize(struct SuperResolutionPtr ptr);

extern "C" superres_API
void SuperResolution_setBlurSigma(struct SuperResolutionPtr ptr, double val);

extern "C" superres_API
double SuperResolution_getBlurSigma(struct SuperResolutionPtr ptr);

extern "C" superres_API
void SuperResolution_setTemporalAreaRadius(struct SuperResolutionPtr ptr, int val);

extern "C" superres_API
int SuperResolution_getTemporalAreaRadius(struct SuperResolutionPtr ptr);

struct DenseOpticalFlowExtPtr {
    void *ptr;

    inline superres::DenseOpticalFlowExt * operator->() { return static_cast<superres::DenseOpticalFlowExt *>(ptr); }
    inline DenseOpticalFlowExtPtr(superres::DenseOpticalFlowExt *ptr) { this->ptr = ptr; }
};

extern "C" superres_API
struct TensorArray DenseOpticalFlowExt_calc(
        struct DenseOpticalFlowExtPtr ptr, struct TensorWrapper frame0, struct TensorWrapper frame1,
        struct TensorWrapper flow1, struct TensorWrapper flow2);

extern "C" superres_API
void DenseOpticalFlowExt_collectGarbage(struct DenseOpticalFlowExtPtr ptr);

extern "C" superres_API
void SuperResolution_setOpticalFlow(struct SuperResolutionPtr ptr, struct DenseOpticalFlowExtPtr val);

extern "C" superres_API
struct DenseOpticalFlowExtPtr SuperResolution_getOpticalFlow(struct SuperResolutionPtr ptr);

struct FarnebackOpticalFlowPtr {
    void *ptr;

    inline superres::FarnebackOpticalFlow * operator->() { return static_cast<superres::FarnebackOpticalFlow *>(ptr); }
    inline FarnebackOpticalFlowPtr(superres::FarnebackOpticalFlow *ptr) { this->ptr = ptr; }
};

struct DualTVL1OpticalFlowPtr {
    void *ptr;

    inline superres::DualTVL1OpticalFlow * operator->() { return static_cast<superres::DualTVL1OpticalFlow *>(ptr); }
    inline DualTVL1OpticalFlowPtr(superres::DualTVL1OpticalFlow *ptr) { this->ptr = ptr; }
};

struct BroxOpticalFlowPtr {
    void *ptr;

    inline superres::BroxOpticalFlow * operator->() { return static_cast<superres::BroxOpticalFlow *>(ptr); }
    inline BroxOpticalFlowPtr(superres::BroxOpticalFlow *ptr) { this->ptr = ptr; }
};

struct PyrLKOpticalFlowPtr {
    void *ptr;

    inline superres::PyrLKOpticalFlow * operator->() { return static_cast<superres::PyrLKOpticalFlow *>(ptr); }
    inline PyrLKOpticalFlowPtr(superres::PyrLKOpticalFlow *ptr) { this->ptr = ptr; }
};

// FarnebackOpticalFlow

extern "C" superres_API
struct FarnebackOpticalFlowPtr createOptFlow_Farneback();

extern "C" superres_API
struct FarnebackOpticalFlowPtr createOptFlow_Farneback_CUDA();

extern "C" superres_API
void FarnebackOpticalFlow_setPyrScale(struct FarnebackOpticalFlowPtr ptr, double val);

extern "C" superres_API
double FarnebackOpticalFlow_getPyrScale(struct FarnebackOpticalFlowPtr ptr);

extern "C" superres_API
void FarnebackOpticalFlow_setLevelsNumber(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C" superres_API
int FarnebackOpticalFlow_getLevelsNumber(struct FarnebackOpticalFlowPtr ptr);

extern "C" superres_API
void FarnebackOpticalFlow_setWindowSize(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C" superres_API
int FarnebackOpticalFlow_getWindowSize(struct FarnebackOpticalFlowPtr ptr);

extern "C" superres_API
void FarnebackOpticalFlow_setIterations(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C" superres_API
int FarnebackOpticalFlow_getIterations(struct FarnebackOpticalFlowPtr ptr);

extern "C" superres_API
void FarnebackOpticalFlow_setPolyN(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C" superres_API
int FarnebackOpticalFlow_getPolyN(struct FarnebackOpticalFlowPtr ptr);

extern "C" superres_API
void FarnebackOpticalFlow_setPolySigma(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C" superres_API
double FarnebackOpticalFlow_getPolySigma(struct FarnebackOpticalFlowPtr ptr);

extern "C" superres_API
void FarnebackOpticalFlow_setFlags(struct FarnebackOpticalFlowPtr ptr, int val);

extern "C" superres_API
int FarnebackOpticalFlow_getFlags(struct FarnebackOpticalFlowPtr ptr);

extern "C" superres_API
struct DualTVL1OpticalFlowPtr createOptFlow_DualTVL1();

extern "C" superres_API
struct DualTVL1OpticalFlowPtr createOptFlow_DualTVL1_CUDA();

extern "C" superres_API
void DualTVL1OpticalFlow_setTau(struct DualTVL1OpticalFlowPtr ptr, double val);

extern "C" superres_API
double DualTVL1OpticalFlow_getTau(struct DualTVL1OpticalFlowPtr ptr);

extern "C" superres_API
void DualTVL1OpticalFlow_setLambda(struct DualTVL1OpticalFlowPtr ptr, double val);

extern "C" superres_API
double DualTVL1OpticalFlow_getLambda(struct DualTVL1OpticalFlowPtr ptr);

extern "C" superres_API
void DualTVL1OpticalFlow_setTheta(struct DualTVL1OpticalFlowPtr ptr, double val);

extern "C" superres_API
double DualTVL1OpticalFlow_getTheta(struct DualTVL1OpticalFlowPtr ptr);

extern "C" superres_API
void DualTVL1OpticalFlow_setScalesNumber(struct DualTVL1OpticalFlowPtr ptr, int val);

extern "C" superres_API
int DualTVL1OpticalFlow_getScalesNumber(struct DualTVL1OpticalFlowPtr ptr);

extern "C" superres_API
void DualTVL1OpticalFlow_setWarpingsNumber(struct DualTVL1OpticalFlowPtr ptr, int val);

extern "C" superres_API
int DualTVL1OpticalFlow_getWarpingsNumber(struct DualTVL1OpticalFlowPtr ptr);

extern "C" superres_API
void DualTVL1OpticalFlow_setEpsilon(struct DualTVL1OpticalFlowPtr ptr, double val);

extern "C" superres_API
double DualTVL1OpticalFlow_getEpsilon(struct DualTVL1OpticalFlowPtr ptr);

extern "C" superres_API
void DualTVL1OpticalFlow_setIterations(struct DualTVL1OpticalFlowPtr ptr, int val);

extern "C" superres_API
int DualTVL1OpticalFlow_getIterations(struct DualTVL1OpticalFlowPtr ptr);

extern "C" superres_API
void DualTVL1OpticalFlow_setUseInitialFlow(struct DualTVL1OpticalFlowPtr ptr, bool val);

extern "C" superres_API
bool DualTVL1OpticalFlow_getUseInitialFlow(struct DualTVL1OpticalFlowPtr ptr);

extern "C" superres_API
struct BroxOpticalFlowPtr createOptFlow_Brox_CUDA();

extern "C" superres_API
void BroxOpticalFlow_setAlpha(struct BroxOpticalFlowPtr ptr, double val);

extern "C" superres_API
double BroxOpticalFlow_getAlpha(struct BroxOpticalFlowPtr ptr);

extern "C" superres_API
void BroxOpticalFlow_setGamma(struct BroxOpticalFlowPtr ptr, double val);

extern "C" superres_API
double BroxOpticalFlow_getGamma(struct BroxOpticalFlowPtr ptr);

extern "C" superres_API
void BroxOpticalFlow_setScaleFactor(struct BroxOpticalFlowPtr ptr, double val);

extern "C" superres_API
double BroxOpticalFlow_getScaleFactor(struct BroxOpticalFlowPtr ptr);

extern "C" superres_API
void BroxOpticalFlow_setInnerIterations(struct BroxOpticalFlowPtr ptr, int val);

extern "C" superres_API
int BroxOpticalFlow_getInnerIterations(struct BroxOpticalFlowPtr ptr);

extern "C" superres_API
void BroxOpticalFlow_setOuterIterations(struct BroxOpticalFlowPtr ptr, int val);

extern "C" superres_API
int BroxOpticalFlow_getOuterIterations(struct BroxOpticalFlowPtr ptr);

extern "C" superres_API
void BroxOpticalFlow_setSolverIterations(struct BroxOpticalFlowPtr ptr, int val);

extern "C" superres_API
int BroxOpticalFlow_getSolverIterations(struct BroxOpticalFlowPtr ptr);

extern "C" superres_API
struct PyrLKOpticalFlowPtr createOptFlow_PyrLK_CUDA();

extern "C" superres_API
void PyrLKOpticalFlow_setWindowSize(struct PyrLKOpticalFlowPtr ptr, int val);

extern "C" superres_API
int PyrLKOpticalFlow_getWindowSize(struct PyrLKOpticalFlowPtr ptr);

extern "C" superres_API
void PyrLKOpticalFlow_setMaxLevel(struct PyrLKOpticalFlowPtr ptr, int val);

extern "C" superres_API
int PyrLKOpticalFlow_getMaxLevel(struct PyrLKOpticalFlowPtr ptr);

extern "C" superres_API
void PyrLKOpticalFlow_setIterations(struct PyrLKOpticalFlowPtr ptr, int val);

extern "C" superres_API
int PyrLKOpticalFlow_getIterations(struct PyrLKOpticalFlowPtr ptr);
