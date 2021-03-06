#include <Common.hpp>
#include <Classes.hpp>
#include <features2d.hpp>
#include <opencv2/calib3d.hpp>

// calib3d_EXPORTS is defined by CMake(add_library)
#if defined (_WIN32) && defined (_MSC_VER)
  #if defined(calib3d_EXPORTS)
    #define  calib3d_API __declspec(dllexport)
  #else
    #define  calib3d_API __declspec(dllimport)
  #endif /* calib3d_EXPORTS */
#else /* defined (_WIN32) */
 #define calib3d_API
#endif

struct calibrateCameraRetval {
    double retval;
    struct TensorArray intrinsics, rvecs, tvecs;
};

struct decomposeHomographyMatRetval {
   int val;
   struct TensorArray rotations, translations, normals;
};

extern "C" calib3d_API
struct calibrateCameraRetval calibrateCamera(
	struct TensorArray objectPoints, struct TensorArray imagePoints,
	struct SizeWrapper imageSize, struct TensorWrapper cameraMatrix,
	struct TensorWrapper distCoeffs, struct TensorArray rvecs,
	struct TensorArray tvecs, int flags, struct TermCriteriaWrapper criteria);

extern "C" calib3d_API
struct TensorWrapper calibrationMatrixValues(
	struct TensorWrapper cameraMatrix,
	struct SizeWrapper imageSize,
	double apertureWidth, double apertureHeight);

extern "C" calib3d_API
struct TensorArray composeRT(
	struct TensorWrapper rvec1, struct TensorWrapper tvec1, struct TensorWrapper rvec2,
	struct TensorWrapper tvec2, struct TensorWrapper rvec3, struct TensorWrapper tvec3,
	struct TensorWrapper dr3dr1, struct TensorWrapper dr3dt1, struct TensorWrapper dr3dr2,
	struct TensorWrapper dr3dt2, struct TensorWrapper dt3dr1, struct TensorWrapper dt3dt1,
	struct TensorWrapper dt3dr2, struct TensorWrapper dt3dt2);

extern "C" calib3d_API
struct TensorWrapper computeCorrespondEpilines(
	struct TensorWrapper points, int whichImage, struct TensorWrapper F,
	struct TensorWrapper lines);

extern "C" calib3d_API
struct TensorWrapper convertPointsFromHomogeneous(
	struct TensorWrapper src, struct TensorWrapper dst);

extern "C" calib3d_API
struct TensorWrapper convertPointsHomogeneous(
	struct TensorWrapper src, struct TensorWrapper dst);

extern "C" calib3d_API
struct TensorWrapper convertPointsToHomogeneous(
	struct TensorWrapper src, struct TensorWrapper dst);

extern "C" calib3d_API
struct TensorArray correctMatches(
	struct TensorWrapper F, struct TensorWrapper points1,
	struct TensorWrapper points2, struct TensorWrapper newPoints1,
	struct TensorWrapper newPoints2);

extern "C" calib3d_API
struct TensorArray decomposeEssentialMat(
	struct TensorWrapper E, struct TensorWrapper R1,
	struct TensorWrapper R2, struct TensorWrapper t);

extern "C" calib3d_API
struct decomposeHomographyMatRetval decomposeHomographyMat(
	struct TensorWrapper H, struct TensorWrapper K,
	struct TensorArray rotations, struct TensorArray translations,
	struct TensorArray normals);

extern "C" calib3d_API
struct TensorArray decomposeProjectionMatrix(
	struct TensorWrapper projMatrix, struct TensorWrapper cameraMatrix,
	struct TensorWrapper rotMatrix, struct TensorWrapper transVect,
	struct TensorWrapper rotMatrixX, struct TensorWrapper rotMatrixY,
	struct TensorWrapper rotMatrixZ, struct TensorWrapper eulerAngles);

extern "C" calib3d_API
void drawChessboardCorners(
	struct TensorWrapper image, struct SizeWrapper patternSize,
	struct TensorWrapper corners, bool patternWasFound);

extern "C" calib3d_API
struct TensorArrayPlusInt estimateAffine3D(
	struct TensorWrapper src, struct TensorWrapper dst,
	struct TensorWrapper out, struct TensorWrapper inliers,
	double ransacThreshold, double confidence);

extern "C" calib3d_API
void filterSpeckles(
	struct TensorWrapper img, double newVal, int maxSpeckleSize,
	double maxDiff, struct TensorWrapper buf);
 
extern "C" calib3d_API
struct TensorWrapper find4QuadCornerSubpix(
	struct TensorWrapper img, struct TensorWrapper corners,
	struct SizeWrapper region_size);

extern "C" calib3d_API
struct TensorPlusBool findChessboardCorners(
	struct TensorWrapper image, struct SizeWrapper patternSize,
	struct TensorWrapper corners, int flags);

extern "C" calib3d_API
struct TensorPlusBool findCirclesGrid(
	struct TensorWrapper image, struct SizeWrapper patternSize,
	struct TensorWrapper centers, int flags, struct SimpleBlobDetectorPtr blobDetector);

extern "C" calib3d_API
struct TensorWrapper findEssentialMat(
	struct TensorWrapper points1, struct TensorWrapper points2,
	double focal, struct Point2dWrapper pp, int method, double prob,
	double threshold, struct TensorWrapper mask);

extern "C" calib3d_API
struct TensorWrapper findFundamentalMat(
	struct TensorWrapper points1, struct TensorWrapper points2, int method,
	double param1, double param2, struct TensorWrapper mask);

extern "C" calib3d_API
struct TensorArray findFundamentalMat2(
	struct TensorWrapper points1, struct TensorWrapper points2,
	struct TensorWrapper mask, int method, double param1, double param2);

extern "C" calib3d_API
struct TensorWrapper findHomography(
	struct TensorWrapper srcPoints, struct TensorWrapper dstPoints,
	int method, double ransacReprojThreshold, struct TensorWrapper mask,
	const int maxIters, const double confidence);

extern "C" calib3d_API
struct TensorArray findHomography2(
	struct TensorWrapper srcPoints, struct TensorWrapper dstPoints,
	struct TensorWrapper mask, int method, double ransacReprojThreshold);

extern "C" calib3d_API
struct TensorPlusRect getOptimalNewCameraMatrix(
	struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
	struct SizeWrapper imageSize, double alpha, struct SizeWrapper newImgSize,
	bool centerPrincipalPoint);

extern "C" calib3d_API
struct RectWrapper getValidDisparityROI(
	struct RectWrapper roi1, struct RectWrapper roi2,
	int minDisparity, int numberOfDisparities, int SADWindowSize);

extern "C" calib3d_API
struct TensorWrapper initCameraMatrix2D(
	struct TensorArray objectPoints, struct TensorArray imagePoints,
   	struct SizeWrapper imageSize, double aspectRatio);

extern "C" calib3d_API
struct TensorArray matMulDeriv(
	struct TensorWrapper A, struct TensorWrapper B,
	struct TensorWrapper dABdA, struct TensorWrapper dABdB);

extern "C" calib3d_API
struct TensorArray projectPoints(
	struct TensorWrapper objectPoints, struct TensorWrapper rvec,
	struct TensorWrapper tvec, struct TensorWrapper cameraMatrix,
	struct TensorWrapper distCoeffs, struct TensorWrapper imagePoints,
	struct TensorWrapper jacobian, double aspectRatio);

extern "C" calib3d_API
struct TensorArrayPlusInt recoverPose(
	struct TensorWrapper E, struct TensorWrapper points1,
	struct TensorWrapper points2, struct TensorWrapper R,
	struct TensorWrapper t, double focal,
	struct Point2dWrapper pp, struct TensorWrapper mask);

extern "C" calib3d_API
struct TensorArrayPlusRectArrayPlusFloat rectify3Collinear(
		struct TensorWrapper cameraMatrix1, struct TensorWrapper distCoeffs1,
		struct TensorWrapper cameraMatrix2, struct TensorWrapper distCoeffs2,
		struct TensorWrapper cameraMatrix3, struct TensorWrapper distCoeffs3,
		struct TensorArray imgpt1, struct TensorArray imgpt3,
		struct SizeWrapper imageSize, struct TensorWrapper R12,
		struct TensorWrapper T12, struct TensorWrapper R13,
		struct TensorWrapper T13, struct TensorWrapper R1,
		struct TensorWrapper R2, struct TensorWrapper R3,
		struct TensorWrapper P1, struct TensorWrapper P2,
		struct TensorWrapper P3, struct TensorWrapper Q,
		double alpha, struct SizeWrapper newImgSize, int flags);

extern "C" calib3d_API
struct TensorWrapper reprojectImageTo3D(
	struct TensorWrapper disparity, struct TensorWrapper _3dImage,
	struct TensorWrapper Q, bool handleMissingValues, int ddepth);

extern "C" calib3d_API
struct TensorArray Rodrigues(
	struct TensorWrapper src, struct TensorWrapper dst, struct TensorWrapper jacobian);

extern "C" calib3d_API
struct TensorArrayPlusVec3d RQDecomp3x3(
	struct TensorWrapper src, struct TensorWrapper mtxR, struct TensorWrapper mtxQ,
	struct TensorWrapper Qx, struct TensorWrapper Qy, struct TensorWrapper Qz);

extern "C" calib3d_API
struct TensorArrayPlusBool solvePnP(
	struct TensorWrapper objectPoints, struct TensorWrapper imagePoints,
	struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
	struct TensorWrapper rvec, struct TensorWrapper tvec,
	bool useExtrinsicGuess, int flags);

extern "C" calib3d_API
struct TensorArrayPlusBool solvePnPRansac(
	struct TensorWrapper objectPoints, struct TensorWrapper imagePoints,
	struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
	struct TensorWrapper rvec, struct TensorWrapper tvec,
	bool useExtrinsicGuess, int iterationsCount, float reprojectionError,
	double confidence, struct TensorWrapper inliers, int flags);

extern "C" calib3d_API
struct TensorArrayPlusDouble stereoCalibrate(
		struct TensorArray objectPoints, struct TensorArray imagePoints1,
		struct TensorArray imagePoints2, struct TensorWrapper cameraMatrix1,
		struct TensorWrapper distCoeffs1, struct TensorWrapper cameraMatrix2,
		struct TensorWrapper distCoeffs2, struct SizeWrapper imageSize,
		struct TensorWrapper R, struct TensorWrapper T,
		struct TensorWrapper E, struct TensorWrapper F,
		int flags, struct TermCriteriaWrapper criteria);

extern "C" calib3d_API
struct TensorArrayPlusRectArray stereoRectify(
	struct TensorWrapper cameraMatrix1, struct TensorWrapper distCoeffs1,
	struct TensorWrapper cameraMatrix2, struct TensorWrapper distCoeffs2,
	struct SizeWrapper imageSize, struct TensorWrapper R,
	struct TensorWrapper T, struct TensorWrapper R1,
	struct TensorWrapper R2, struct TensorWrapper P1,
	struct TensorWrapper P2, struct TensorWrapper Q,
	int flags, double alpha, struct SizeWrapper newImageSize);

extern "C" calib3d_API
struct TensorArrayPlusBool stereoRectifyUncalibrated(
	struct TensorWrapper points1, struct TensorWrapper points2,
	struct TensorWrapper F, struct SizeWrapper imgSize,
	struct TensorWrapper H1, struct TensorWrapper H2, double threshold);

extern "C" calib3d_API
struct TensorWrapper triangulatePoints(
	struct TensorWrapper projMatr1, struct TensorWrapper projMatr2,
	struct TensorWrapper projPoints1, struct TensorWrapper projPoints2,
	struct TensorWrapper points4D);

extern "C" calib3d_API
struct TensorWrapper validateDisparity(
	struct TensorWrapper disparity, struct TensorWrapper cost,
        int minDisparity, int numberOfDisparities, int disp12MaxDisp);

//******************Fisheye camera model***************

namespace fisheye = cv::fisheye;

extern "C" calib3d_API
struct calibrateCameraRetval fisheye_calibrate(
	struct TensorArray objectPoints, struct TensorArray imagePoints,
	struct SizeWrapper imageSize, struct TensorWrapper K,
	struct TensorWrapper D, struct TensorArray rvecs,
	struct TensorArray tvecs, int flags, struct TermCriteriaWrapper criteria);

extern "C" calib3d_API
struct TensorWrapper fisheye_distortPoints(
	struct TensorWrapper undistorted, struct TensorWrapper distorted,
	struct TensorWrapper K, struct TensorWrapper D, double alpha);

extern "C" calib3d_API
struct TensorWrapper fisheye_estimateNewCameraMatrixForUndistortRectify(
	struct TensorWrapper K, struct TensorWrapper D,
	struct SizeWrapper image_size, struct TensorWrapper R,
	struct TensorWrapper P, double balance,
	struct SizeWrapper new_size, double fov_scale);

extern "C" calib3d_API
struct TensorArray fisheye_initUndistortRectifyMap(
	struct TensorWrapper K, struct TensorWrapper D,
	struct TensorWrapper R, struct TensorWrapper P,
	struct SizeWrapper size, int m1type,
	struct TensorWrapper map1, struct TensorWrapper map2);

extern "C" calib3d_API
struct TensorArray fisheye_projectPoints2(
	struct TensorWrapper objectPoints, struct TensorWrapper imagePoints,
	struct TensorWrapper rvec, struct TensorWrapper tvec,
	struct TensorWrapper K, struct TensorWrapper D, double alpha,
	struct TensorWrapper jacobian);

extern "C" calib3d_API
struct TensorArrayPlusDouble fisheye_stereoCalibrate(
		struct TensorArray objectPoints, struct TensorArray imagePoints1,
		struct TensorArray imagePoints2, struct TensorWrapper K1,
		struct TensorWrapper D1, struct TensorWrapper K2,
		struct TensorWrapper D2, struct SizeWrapper imageSize,
		struct TensorWrapper R, struct TensorWrapper T,
		int flags, struct TermCriteriaWrapper criteria);

extern "C" calib3d_API
struct TensorArray fisheye_stereoRectify(
	struct TensorWrapper K1, struct TensorWrapper D1,
	struct TensorWrapper K2, struct TensorWrapper D2,
	struct SizeWrapper imageSize, struct TensorWrapper R,
	struct TensorWrapper tvec, struct TensorWrapper R1,
	struct TensorWrapper R2, struct TensorWrapper P1,
	struct TensorWrapper P2, struct TensorWrapper Q,
	int flags, struct SizeWrapper newImageSize,
	double balance, double fov_scale);

extern "C" calib3d_API
struct TensorWrapper fisheye_undistortImage(
	struct TensorWrapper distorted, struct TensorWrapper undistorted,
	struct TensorWrapper K, struct TensorWrapper D,
	struct TensorWrapper Knew, struct SizeWrapper new_size);

extern "C" calib3d_API
struct TensorWrapper fisheye_undistortPoints(
	struct TensorWrapper distorted, struct TensorWrapper undistorted,
	struct TensorWrapper K, struct TensorWrapper D,
	struct TensorWrapper R, struct TensorWrapper P);

/****************** Classes ******************/

//StereoMatcher

extern "C"
struct StereoMatcherPtr {
    void *ptr;

    inline cv::StereoMatcher * operator->() {
			return static_cast<cv::StereoMatcher *>(ptr); }
    inline StereoMatcherPtr(cv::StereoMatcher *ptr) { this->ptr = ptr; }
};

extern "C" calib3d_API
struct TensorWrapper StereoMatcher_compute(
	struct StereoMatcherPtr ptr, struct TensorWrapper left,
	struct TensorWrapper right, struct TensorWrapper disparity);

extern "C" calib3d_API
int StereoMatcher_getBlockSize(
	struct StereoMatcherPtr ptr);

extern "C" calib3d_API
int StereoMatcher_getDisp12MaxDiff(
	struct StereoMatcherPtr ptr);

extern "C" calib3d_API
int StereoMatcher_getMinDisparity(
	struct StereoMatcherPtr ptr);

extern "C" calib3d_API
int StereoMatcher_getNumDisparities(
	struct StereoMatcherPtr ptr);

extern "C" calib3d_API
int StereoMatcher_getSpeckleRange(
	struct StereoMatcherPtr ptr);

extern "C" calib3d_API
int StereoMatcher_getSpeckleWindowSize(
	struct StereoMatcherPtr ptr);

extern "C" calib3d_API
void StereoMatcher_setBlockSize(
	struct StereoMatcherPtr ptr, int blockSize);

extern "C" calib3d_API
void StereoMatcher_setDisp12MaxDiff(
	struct StereoMatcherPtr ptr, int disp12MaxDiff);

extern "C" calib3d_API
void StereoMatcher_setMinDisparity(
	struct StereoMatcherPtr ptr, int minDisparity);

extern "C" calib3d_API
void StereoMatcher_setNumDisparities(
	struct StereoMatcherPtr ptr, int numDisparities);

extern "C" calib3d_API
void StereoMatcher_setSpeckleRange(
	struct StereoMatcherPtr ptr, int speckleRange);

extern "C" calib3d_API
void StereoMatcher_setSpeckleWindowSize(
	struct StereoMatcherPtr ptr, int speckleWindowSize);


//StereoBM

struct StereoBMPtr {
    void *ptr;

    inline cv::StereoBM * operator->(){
        return static_cast<cv::StereoBM *>(ptr);
    }

    inline StereoBMPtr(cv::StereoBM *ptr){
        this->ptr = ptr;
    }
};

extern "C" calib3d_API
struct StereoBMPtr StereoBM_ctor(
	int numDisparities, int blockSize);

extern "C" calib3d_API
int StereoBM_getPreFilterCap(
	struct StereoBMPtr ptr);

extern "C" calib3d_API
int StereoBM_getPreFilterSize(
	struct StereoBMPtr ptr);

extern "C" calib3d_API
int StereoBM_getPreFilterType(
	struct StereoBMPtr ptr);

extern "C" calib3d_API
struct RectWrapper StereoBM_getROI1(
	struct StereoBMPtr ptr);

extern "C" calib3d_API
struct RectWrapper StereoBM_getROI2(
	struct StereoBMPtr ptr);

extern "C" calib3d_API
int StereoBM_getSmallerBlockSize(
	struct StereoBMPtr ptr);

extern "C" calib3d_API
int StereoBM_getTextureThreshold(
	struct StereoBMPtr ptr);

extern "C" calib3d_API
int StereoBM_getUniquenessRatio(
	struct StereoBMPtr ptr);

extern "C" calib3d_API
void StereoBM_setPreFilterCap(
	struct StereoBMPtr ptr, int preFilterCap);

extern "C" calib3d_API
void StereoBM_setPreFilterSize(
	struct StereoBMPtr ptr, int preFilterSize);

extern "C" calib3d_API
void StereoBM_setPreFilterType(
	struct StereoBMPtr ptr, int preFilterType);

extern "C" calib3d_API
void StereoBM_setROI1(
	struct StereoBMPtr ptr, struct RectWrapper roi1);

extern "C" calib3d_API
void StereoBM_setROI2(
	struct StereoBMPtr ptr, struct RectWrapper roi2);

extern "C" calib3d_API
void StereoBM_setSmallerBlockSize(
	struct StereoBMPtr ptr, int blockSize);

extern "C" calib3d_API
void StereoBM_setTextureThreshold(
	struct StereoBMPtr ptr, int textureThreshold);

extern "C" calib3d_API
void StereoBM_setUniquenessRatio(
	struct StereoBMPtr ptr, int uniquenessRatio);

//StereoSGBM

extern "C"
struct StereoSGBMPtr {
    void *ptr;

    inline cv::StereoSGBM * operator->(){
        return static_cast<cv::StereoSGBM *>(ptr);
    }

    inline StereoSGBMPtr(cv::StereoSGBM *ptr){ 
        this->ptr = ptr;
    }
};

extern "C" calib3d_API
struct StereoSGBMPtr StereoSGBM_ctor(
	int minDisparity, int numDisparities, int blockSize,
	int P1, int P2, int disp12MaxDiff, int preFilterCap,
	int uniquenessRatio, int speckleWindowSize,
	int speckleRange, int mode);

extern "C" calib3d_API
int StereoSGBM_getMode(
	struct StereoSGBMPtr ptr);

extern "C" calib3d_API
int StereoSGBM_getP1(
	struct StereoSGBMPtr ptr);

extern "C" calib3d_API
int StereoSGBM_getP2(
	struct StereoSGBMPtr ptr);

extern "C" calib3d_API
int StereoSGBM_getPreFilterCap(
	struct StereoSGBMPtr ptr);

extern "C" calib3d_API
int StereoSGBM_getUniquenessRatio(
	struct StereoSGBMPtr ptr);

extern "C" calib3d_API
void StereoSGBM_setMode(
	struct StereoSGBMPtr ptr, int mode);

extern "C" calib3d_API
void StereoSGBM_setP1(
	struct StereoSGBMPtr ptr, int P1);

extern "C" calib3d_API
void StereoSGBM_setP2(
	struct StereoSGBMPtr ptr, int P2);

extern "C" calib3d_API
void StereoSGBM_setPreFilterCap(
	struct StereoSGBMPtr ptr, int preFilterCap);

extern "C" calib3d_API
void StereoSGBM_setUniquenessRatio(
	struct StereoSGBMPtr ptr, int uniquenessRatio);




