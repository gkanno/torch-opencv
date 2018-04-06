#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/imgproc.hpp>

// imgproc_EXPORTS is defined by CMake(add_library)
#if defined (_WIN32) && defined (_MSC_VER)
  #if defined(imgproc_EXPORTS)
    #define  imgproc_API __declspec(dllexport)
  #else
    #define  imgproc_API __declspec(dllimport)
  #endif /* imgproc_EXPORTS */
#else /* defined (_WIN32) */
 #define imgproc_API
#endif

extern "C" imgproc_API
struct TensorWrapper getGaussianKernel(int ksize, double sigma, int ktype);

extern "C" imgproc_API
struct TensorArray getDerivKernels(
        int dx, int dy, int ksize, struct TensorWrapper kx,
        struct TensorWrapper ky, bool normalize, int ktype);

extern "C" imgproc_API
struct TensorWrapper getGaborKernel(struct SizeWrapper ksize, double sigma, double theta,
                                               double lambd, double gamma, double psi, int ktype);

extern "C" imgproc_API
struct TensorWrapper getStructuringElement(int shape, struct SizeWrapper ksize,
                                                      struct PointWrapper anchor);

extern "C" imgproc_API
struct TensorWrapper medianBlur(struct TensorWrapper src, int ksize, struct TensorWrapper dst);

extern "C" imgproc_API
struct TensorWrapper GaussianBlur(struct TensorWrapper src, struct SizeWrapper ksize,
                                  double sigmaX, struct TensorWrapper dst,
                                  double sigmaY, int borderType);

extern "C" imgproc_API
struct TensorWrapper bilateralFilter(struct TensorWrapper src, int d,
                                     double sigmaColor, double sigmaSpace,
                                     struct TensorWrapper dst, int borderType);

extern "C" imgproc_API
struct TensorWrapper boxFilter(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct SizeWrapper ksize, struct PointWrapper anchor,
        bool normalize, int borderType);

extern "C" imgproc_API
struct TensorWrapper sqrBoxFilter(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct SizeWrapper ksize, struct PointWrapper anchor,
        bool normalize, int borderType);

extern "C" imgproc_API
struct TensorWrapper blur(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper ksize, struct PointWrapper anchor, int borderType);

extern "C" imgproc_API
struct TensorWrapper filter2D(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct TensorWrapper kernel, struct PointWrapper anchor,
        double delta, int borderType);

extern "C" imgproc_API
struct TensorWrapper sepFilter2D(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct TensorWrapper kernelX,struct TensorWrapper kernelY,
        struct PointWrapper anchor, double delta, int borderType);

extern "C" imgproc_API
struct TensorWrapper Sobel(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int dx, int dy, int ksize, double scale, double delta, int borderType);

extern "C" imgproc_API
struct TensorWrapper Scharr(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int dx, int dy, double scale, double delta, int borderType);

extern "C" imgproc_API
struct TensorWrapper Laplacian(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int ksize, double scale, double delta, int borderType);

extern "C" imgproc_API
struct TensorWrapper Canny(
        struct TensorWrapper image, struct TensorWrapper edges,
        double threshold1, double threshold2, int apertureSize, bool L2gradient);

extern "C" imgproc_API
struct TensorWrapper cornerMinEigenVal(
        struct TensorWrapper src, struct TensorWrapper dst,
        int blockSize, int ksize, int borderType);

extern "C" imgproc_API
struct TensorWrapper cornerHarris(
        struct TensorWrapper src, struct TensorWrapper dst, int blockSize,
        int ksize, double k, int borderType);

extern "C" imgproc_API
struct TensorWrapper cornerEigenValsAndVecs(
        struct TensorWrapper src, struct TensorWrapper dst,
        int blockSize, int ksize, int borderType);

extern "C" imgproc_API
struct TensorWrapper preCornerDetect(
        struct TensorWrapper src, struct TensorWrapper dst, int ksize, int borderType);

extern "C" imgproc_API
struct TensorWrapper HoughLines(
        struct TensorWrapper image,
        double rho, double theta, int threshold, double srn, double stn,
        double min_theta, double max_theta);

extern "C" imgproc_API
struct TensorWrapper HoughLinesP(
        struct TensorWrapper image, double rho,
        double theta, int threshold, double minLineLength, double maxLineGap);

extern "C" imgproc_API
struct TensorWrapper HoughCircles(
        struct TensorWrapper image,
        int method, double dp, double minDist, double param1, double param2,
        int minRadius, int maxRadius);

extern "C" imgproc_API
void cornerSubPix(
        struct TensorWrapper image, struct TensorWrapper corners,
        struct SizeWrapper winSize, struct SizeWrapper zeroZone,
        struct TermCriteriaWrapper criteria);

extern "C" imgproc_API
struct TensorWrapper goodFeaturesToTrack(
        struct TensorWrapper image,
        int maxCorners, double qualityLevel, double minDistance,
        struct TensorWrapper mask, int blockSize, bool useHarrisDetector, double k);

extern "C" imgproc_API
struct ScalarWrapper morphologyDefaultBorderValue();

extern "C" imgproc_API
struct TensorWrapper erode(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper kernel, struct PointWrapper anchor,
        int iterations, int borderType, struct ScalarWrapper borderValue);

extern "C" imgproc_API
struct TensorWrapper dilate(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper kernel, struct PointWrapper anchor,
        int iterations, int borderType, struct ScalarWrapper borderValue);

extern "C" imgproc_API
struct TensorWrapper morphologyEx(
        struct TensorWrapper src, struct TensorWrapper dst,
        int op, struct TensorWrapper kernel, struct PointWrapper anchor,
        int iterations, int borderType, struct ScalarWrapper borderValue);

extern "C" imgproc_API
struct TensorWrapper resize(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dsize, double fx, double fy,
        int interpolation);

extern "C" imgproc_API
struct TensorWrapper warpAffine(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper M, struct SizeWrapper dsize,
        int flags, int borderMode, struct ScalarWrapper borderValue);

extern "C" imgproc_API
struct TensorWrapper warpPerspective(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper M, struct SizeWrapper dsize,
        int flags, int borderMode, struct ScalarWrapper borderValue);

extern "C" imgproc_API
struct TensorWrapper remap(
        struct TensorWrapper src, struct TensorWrapper map1,
        struct TensorWrapper map2, int interpolation, struct TensorWrapper dst,
        int borderMode, struct ScalarWrapper borderValue);

extern "C" imgproc_API
struct TensorArray convertMaps(
        struct TensorWrapper map1, struct TensorWrapper map2,
        struct TensorWrapper dstmap1, struct TensorWrapper dstmap2,
        int dstmap1type, bool nninterpolation);

extern "C" imgproc_API
struct TensorWrapper getRotationMatrix2D(
        struct Point2fWrapper center, double angle, double scale);

extern "C" imgproc_API
struct TensorWrapper invertAffineTransform(
        struct TensorWrapper M, struct TensorWrapper iM);

extern "C" imgproc_API
struct TensorWrapper getPerspectiveTransform(
        struct TensorWrapper src, struct TensorWrapper dst);

extern "C" imgproc_API
struct TensorWrapper getAffineTransform(
        struct TensorWrapper src, struct TensorWrapper dst);

extern "C" imgproc_API
struct TensorWrapper getRectSubPix(
        struct TensorWrapper image, struct SizeWrapper patchSize,
        struct Point2fWrapper center, struct TensorWrapper patch, int patchType);

extern "C" imgproc_API
struct TensorWrapper logPolar(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct Point2fWrapper center, double M, int flags);

extern "C" imgproc_API
struct TensorWrapper linearPolar(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct Point2fWrapper center, double maxRadius, int flags);

extern "C" imgproc_API
struct TensorWrapper integral(
        struct TensorWrapper src, struct TensorWrapper sum, int sdepth);

extern "C" imgproc_API
struct TensorArray integralN(
        struct TensorWrapper src, struct TensorArray sums, int sdepth, int sqdepth);

extern "C" imgproc_API
void accumulate(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper mask);

extern "C" imgproc_API
void accumulateSquare(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper mask);

extern "C" imgproc_API
void accumulateProduct(
        struct TensorWrapper src1, struct TensorWrapper src2,
        struct TensorWrapper dst, struct TensorWrapper mask);

extern "C" imgproc_API
void accumulateWeighted(
        struct TensorWrapper src, struct TensorWrapper dst,
        double alpha, struct TensorWrapper mask);

extern "C" imgproc_API
struct Vec3dWrapper phaseCorrelate(
        struct TensorWrapper src1, struct TensorWrapper src2,
        struct TensorWrapper window);

extern "C" imgproc_API
struct TensorWrapper createHanningWindow(
        struct TensorWrapper dst, struct SizeWrapper winSize, int type);

extern "C" imgproc_API
struct TensorPlusDouble threshold(
        struct TensorWrapper src, struct TensorWrapper dst,
        double thresh, double maxval, int type);

extern "C" imgproc_API
struct TensorWrapper adaptiveThreshold(
        struct TensorWrapper src, struct TensorWrapper dst,
        double maxValue, int adaptiveMethod, int thresholdType,
        int blockSize, double C);

extern "C" imgproc_API
struct TensorWrapper pyrDown(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dstSize, int borderType);

extern "C" imgproc_API
struct TensorWrapper pyrUp(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dstSize, int borderType);

extern "C" imgproc_API
struct TensorArray buildPyramid(
        struct TensorWrapper src, struct TensorArray dst,
        int maxlevel, int borderType);

extern "C" imgproc_API
struct TensorWrapper undistort(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct TensorWrapper newCameraMatrix);

extern "C" imgproc_API
struct TensorArray initUndistortRectifyMap(
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct TensorWrapper R, struct TensorWrapper newCameraMatrix,
        struct SizeWrapper size, int m1type,
        struct TensorArray maps);

extern "C" imgproc_API
struct TensorArrayPlusFloat initWideAngleProjMap(
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct SizeWrapper imageSize, int destImageWidth,
        int m1type, struct TensorArray maps,
        int projType, double alpha);

extern "C" imgproc_API
struct TensorWrapper getDefaultNewCameraMatrix(
        struct TensorWrapper cameraMatrix, struct SizeWrapper imgsize, bool centerPrincipalPoint);

extern "C" imgproc_API
struct TensorWrapper undistortPoints(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct TensorWrapper R, struct TensorWrapper P);

extern "C" imgproc_API
struct TensorWrapper calcHist(
        struct TensorArray images,
        struct TensorWrapper channels, struct TensorWrapper mask,
        struct TensorWrapper hist, int dims, struct TensorWrapper histSize,
        struct TensorWrapper ranges, bool uniform, bool accumulate);

extern "C" imgproc_API
struct TensorWrapper calcBackProject(
        struct TensorArray images, int nimages,
        struct TensorWrapper channels, struct TensorWrapper hist,
        struct TensorWrapper backProject, struct TensorWrapper ranges,
        double scale, bool uniform);

extern "C" imgproc_API
double compareHist(
        struct TensorWrapper H1, struct TensorWrapper H2, int method);

extern "C" imgproc_API
struct TensorWrapper equalizeHist(
        struct TensorWrapper src, struct TensorWrapper dst);

extern "C" imgproc_API
struct TensorPlusFloat EMD(
        struct TensorWrapper signature1, struct TensorWrapper signature2,
        int distType, struct TensorWrapper cost,
        struct TensorWrapper lowerBound, struct TensorWrapper flow);

extern "C" imgproc_API
void watershed(
        struct TensorWrapper image, struct TensorWrapper markers);

extern "C" imgproc_API
struct TensorWrapper pyrMeanShiftFiltering(
        struct TensorWrapper src, struct TensorWrapper dst,
        double sp, double sr, int maxLevel, struct TermCriteriaWrapper termcrit);

extern "C" imgproc_API
struct TensorArray grabCut(
        struct TensorWrapper img, struct TensorWrapper mask,
        struct RectWrapper rect, struct TensorWrapper bgdModel,
        struct TensorWrapper fgdModel, int iterCount, int mode);

extern "C" imgproc_API
struct TensorWrapper distanceTransform(
        struct TensorWrapper src, struct TensorWrapper dst,
        int distanceType, int maskSize, int dstType);

extern "C" imgproc_API
struct TensorArray distanceTransformWithLabels(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper labels, int distanceType, int maskSize,
        int labelType);

extern "C" imgproc_API
struct RectPlusInt floodFill(
        struct TensorWrapper image, struct TensorWrapper mask,
        struct PointWrapper seedPoint, struct ScalarWrapper newVal,
        struct ScalarWrapper loDiff, struct ScalarWrapper upDiff, int flags);

extern "C" imgproc_API
struct TensorWrapper cvtColor(
        struct TensorWrapper src, struct TensorWrapper dst, int code, int dstCn);

extern "C" imgproc_API
struct TensorWrapper demosaicing(
        struct TensorWrapper _src, struct TensorWrapper _dst, int code, int dcn);

extern "C" imgproc_API
struct MomentsWrapper moments(
        struct TensorWrapper array, bool binaryImage);

extern "C" imgproc_API
struct TensorWrapper HuMoments(
        struct MomentsWrapper m);

extern "C" imgproc_API
struct TensorWrapper matchTemplate(
        struct TensorWrapper image, struct TensorWrapper templ, struct TensorWrapper result, int method, struct TensorWrapper mask);

extern "C" imgproc_API
struct TensorPlusInt connectedComponents(
        struct TensorWrapper image, struct TensorWrapper labels, int connectivity, int ltype);

extern "C" imgproc_API
struct TensorArrayPlusInt connectedComponentsWithStats(
        struct TensorWrapper image, struct TensorArray outputTensors, int connectivity, int ltype);

extern "C" imgproc_API
struct TensorArray findContours(
        struct TensorWrapper image, bool withHierarchy, int mode, int method, struct PointWrapper offset);

extern "C" imgproc_API
struct TensorWrapper approxPolyDP(
        struct TensorWrapper curve, struct TensorWrapper approxCurve, double epsilon, bool closed);

extern "C" imgproc_API
double arcLength(
        struct TensorWrapper curve, bool closed);

extern "C" imgproc_API
struct RectWrapper boundingRect(
        struct TensorWrapper points);

extern "C" imgproc_API
double contourArea(
        struct TensorWrapper contour, bool oriented);

extern "C" imgproc_API
struct RotatedRectWrapper minAreaRect(
        struct TensorWrapper points);

extern "C" imgproc_API
struct TensorWrapper boxPoints(
        struct RotatedRectWrapper box, struct TensorWrapper points);

extern "C" imgproc_API
struct Vec3fWrapper minEnclosingCircle(
        struct TensorWrapper points, struct Point2fWrapper center, float radius);

extern "C" imgproc_API
struct TensorPlusDouble minEnclosingTriangle(
        struct TensorWrapper points, struct TensorWrapper triangle);

extern "C" imgproc_API
double matchShapes(
        struct TensorWrapper contour1, struct TensorWrapper contour2, int method, double parameter);

extern "C" imgproc_API
struct TensorWrapper convexHull(
        struct TensorWrapper points, struct TensorWrapper hull,
        bool clockwise, bool returnPoints);

extern "C" imgproc_API
struct TensorWrapper convexityDefects(
        struct TensorWrapper contour, struct TensorWrapper convexhull,
        struct TensorWrapper convexityDefects);

extern "C" imgproc_API
bool isContourConvex(
        struct TensorWrapper contour);

extern "C" imgproc_API
struct TensorPlusFloat intersectConvexConvex(
        struct TensorWrapper _p1, struct TensorWrapper _p2,
        struct TensorWrapper _p12, bool handleNested);

extern "C" imgproc_API
struct RotatedRectWrapper fitEllipse(
        struct TensorWrapper points);

extern "C" imgproc_API
struct TensorWrapper fitLine(
        struct TensorWrapper points, struct TensorWrapper line, int distType,
        double param, double reps, double aeps);

extern "C" imgproc_API
double pointPolygonTest(
        struct TensorWrapper contour, struct Point2fWrapper pt, bool measureDist);

extern "C" imgproc_API
struct TensorWrapper rotatedRectangleIntersection(
        struct RotatedRectWrapper rect1, struct RotatedRectWrapper rect2);

extern "C" imgproc_API
struct TensorWrapper blendLinear(
        struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper weights1, struct TensorWrapper weights2, struct TensorWrapper dst);

extern "C" imgproc_API
struct TensorWrapper applyColorMap(
        struct TensorWrapper src, struct TensorWrapper dst, int colormap);

extern "C" imgproc_API
void line(
        struct TensorWrapper img, struct PointWrapper pt1, struct PointWrapper pt2, struct ScalarWrapper color, int thickness, int lineType, int shift);

extern "C" imgproc_API
void arrowedLine(
        struct TensorWrapper img, struct PointWrapper pt1, struct PointWrapper pt2, struct ScalarWrapper color, int thickness, int line_type, int shift, double tipLength);

extern "C" imgproc_API
void rectangle(
        struct TensorWrapper img, struct PointWrapper pt1, struct PointWrapper pt2, struct ScalarWrapper color, int thickness, int lineType, int shift);

extern "C" imgproc_API
void rectangle2(
        struct TensorWrapper img, struct RectWrapper rec, struct ScalarWrapper color, int thickness, int lineType, int shift);

extern "C" imgproc_API
void circle(
        struct TensorWrapper img, struct PointWrapper center, int radius, struct ScalarWrapper color, int thickness, int lineType, int shift);

extern "C" imgproc_API
void ellipse(
        struct TensorWrapper img, struct PointWrapper center, struct SizeWrapper axes, double angle, double startAngle, double endAngle, struct ScalarWrapper color, int thickness, int lineType, int shift);

extern "C" imgproc_API
void ellipseFromRect(
        struct TensorWrapper img, struct RotatedRectWrapper box, struct ScalarWrapper color, int thickness, int lineType);

extern "C" imgproc_API
void fillConvexPoly(
        struct TensorWrapper img, struct TensorWrapper points, struct ScalarWrapper color, int lineType, int shift);

extern "C" imgproc_API
void fillPoly(
        struct TensorWrapper img, struct TensorArray pts, struct ScalarWrapper color, int lineType, int shift, struct PointWrapper offset);

extern "C" imgproc_API
void polylines(
        struct TensorWrapper img, struct TensorArray pts, bool isClosed, struct ScalarWrapper color, int thickness, int lineType, int shift);

extern "C" imgproc_API
void drawContours(
        struct TensorWrapper image, struct TensorArray contours, int contourIdx, struct ScalarWrapper color, int thickness, int lineType, struct TensorWrapper hierarchy, int maxLevel, struct PointWrapper offset);

extern "C" imgproc_API
struct ScalarPlusBool clipLineSize(
        struct SizeWrapper imgSize, struct PointWrapper pt1, struct PointWrapper pt2);

extern "C" imgproc_API
struct ScalarPlusBool clipLineRect(
        struct RectWrapper imgRect, struct PointWrapper pt1, struct PointWrapper pt2);

extern "C" imgproc_API
struct TensorWrapper ellipse2Poly(
        struct PointWrapper center, struct SizeWrapper axes, int angle, int arcStart, int arcEnd, int delta);

extern "C" imgproc_API
void putText(
        struct TensorWrapper img, const char *text, struct PointWrapper org, int fontFace, double fontScale, struct ScalarWrapper color, int thickness, int lineType, bool bottomLeftOrigin);

extern "C" imgproc_API
struct SizePlusInt getTextSize(
        const char *text, int fontFace, double fontScale, int thickness);

struct GeneralizedHoughPtr {
    void *ptr;

    inline cv::GeneralizedHough * operator->() { return static_cast<cv::GeneralizedHough *>(ptr); }
    inline GeneralizedHoughPtr(cv::GeneralizedHough *ptr) { this->ptr = ptr; }
};

struct GeneralizedHoughBallardPtr {
    void *ptr;

    inline cv::GeneralizedHoughBallard * operator->() { return static_cast<cv::GeneralizedHoughBallard *>(ptr); }
    inline GeneralizedHoughBallardPtr(cv::GeneralizedHoughBallard *ptr) { this->ptr = ptr; }
};

struct GeneralizedHoughGuilPtr {
    void *ptr;

    inline cv::GeneralizedHoughGuil * operator->() { return static_cast<cv::GeneralizedHoughGuil *>(ptr); }
    inline GeneralizedHoughGuilPtr(cv::GeneralizedHoughGuil *ptr) { this->ptr = ptr; }
};

struct CLAHEPtr {
    void *ptr;

    inline cv::CLAHE * operator->() { return static_cast<cv::CLAHE *>(ptr); }
    inline CLAHEPtr(cv::CLAHE *ptr) { this->ptr = ptr; }
};

struct LineSegmentDetectorPtr {
    void *ptr;

    inline cv::LineSegmentDetector * operator->() { return static_cast<cv::LineSegmentDetector *>(ptr); }
    inline LineSegmentDetectorPtr(cv::LineSegmentDetector *ptr) { this->ptr = ptr; }
};

struct Subdiv2DPtr {
    void *ptr;

    inline cv::Subdiv2D * operator->() { return static_cast<cv::Subdiv2D *>(ptr); }
    inline Subdiv2DPtr(cv::Subdiv2D *ptr) { this->ptr = ptr; }
};

struct LineIteratorPtr {
    void *ptr;

    inline cv::LineIterator * operator->() { return static_cast<cv::LineIterator *>(ptr); }
    inline LineIteratorPtr(cv::LineIterator *ptr) { this->ptr = ptr; }
};

extern "C" imgproc_API
void GeneralizedHough_setTemplate(
        GeneralizedHoughPtr ptr, struct TensorWrapper templ, struct PointWrapper templCenter);

extern "C" imgproc_API
void GeneralizedHough_setTemplate_edges(
        GeneralizedHoughPtr ptr, struct TensorWrapper edges, struct TensorWrapper dx,
        struct TensorWrapper dy, struct PointWrapper templCenter);

extern "C" imgproc_API
struct TensorArray GeneralizedHough_detect(
        GeneralizedHoughPtr ptr, struct TensorWrapper image, struct TensorWrapper positions, bool votes);

extern "C" imgproc_API
struct TensorArray GeneralizedHough_detect_edges(
        GeneralizedHoughPtr ptr, struct TensorWrapper edges, struct TensorWrapper dx,
        struct TensorWrapper dy, struct TensorWrapper positions, bool votes);

extern "C" imgproc_API
void GeneralizedHough_setCannyLowThresh(GeneralizedHoughPtr ptr, int cannyLowThresh);

extern "C" imgproc_API
int GeneralizedHough_getCannyLowThresh(GeneralizedHoughPtr ptr);

extern "C" imgproc_API
void GeneralizedHough_setCannyHighThresh(GeneralizedHoughPtr ptr, int cannyHighThresh);

extern "C" imgproc_API
int GeneralizedHough_getCannyHighThresh(GeneralizedHoughPtr ptr);

extern "C" imgproc_API
void GeneralizedHough_setMinDist(GeneralizedHoughPtr ptr, double MinDist);

extern "C" imgproc_API
double GeneralizedHough_getMinDist(GeneralizedHoughPtr ptr);

extern "C" imgproc_API
void GeneralizedHough_setDp(GeneralizedHoughPtr ptr, double Dp);

extern "C" imgproc_API
double GeneralizedHough_getDp(GeneralizedHoughPtr ptr);

extern "C" imgproc_API
void GeneralizedHough_setMaxBufferSize(GeneralizedHoughPtr ptr, int MaxBufferSize);

extern "C" imgproc_API
int GeneralizedHough_getMaxBufferSize(GeneralizedHoughPtr ptr);

extern "C" imgproc_API
struct GeneralizedHoughBallardPtr GeneralizedHoughBallard_ctor();

extern "C" imgproc_API
void GeneralizedHoughBallard_setLevels(GeneralizedHoughBallardPtr ptr, double Levels);

extern "C" imgproc_API
double GeneralizedHoughBallard_getLevels(GeneralizedHoughBallardPtr ptr);

extern "C" imgproc_API
void GeneralizedHoughBallard_setVotesThreshold(GeneralizedHoughBallardPtr ptr, double votesThreshold);

extern "C" imgproc_API
double GeneralizedHoughBallard_getVotesThreshold(GeneralizedHoughBallardPtr ptr);

extern "C" imgproc_API
struct GeneralizedHoughGuilPtr GeneralizedHoughGuil_ctor();

extern "C" imgproc_API
void GeneralizedHoughGuil_setLevels(GeneralizedHoughGuilPtr ptr, int levels);

extern "C" imgproc_API
int GeneralizedHoughGuil_getLevels(GeneralizedHoughGuilPtr ptr);

extern "C" imgproc_API
void GeneralizedHoughGuil_setAngleEpsilon(GeneralizedHoughGuilPtr ptr, double AngleEpsilon);

extern "C" imgproc_API
double GeneralizedHoughGuil_getAngleEpsilon(GeneralizedHoughGuilPtr ptr);

extern "C" imgproc_API
void GeneralizedHoughGuil_setMinAngle(GeneralizedHoughGuilPtr ptr, double MinAngle);

extern "C" imgproc_API
double GeneralizedHoughGuil_getMinAngle(GeneralizedHoughGuilPtr ptr);

extern "C" imgproc_API
void GeneralizedHoughGuil_setMaxAngle(GeneralizedHoughGuilPtr ptr, double MaxAngle);

extern "C" imgproc_API
double GeneralizedHoughGuil_getMaxAngle(GeneralizedHoughGuilPtr ptr);

extern "C" imgproc_API
void GeneralizedHoughGuil_setAngleStep(GeneralizedHoughGuilPtr ptr, double AngleStep);

extern "C" imgproc_API
double GeneralizedHoughGuil_getAngleStep(GeneralizedHoughGuilPtr ptr);

extern "C" imgproc_API
void GeneralizedHoughGuil_setAngleThresh(GeneralizedHoughGuilPtr ptr, int AngleThresh);

extern "C" imgproc_API
int GeneralizedHoughGuil_getAngleThresh(GeneralizedHoughGuilPtr ptr);

extern "C" imgproc_API
void GeneralizedHoughGuil_setMinScale(GeneralizedHoughGuilPtr ptr, double MinScale);

extern "C" imgproc_API
double GeneralizedHoughGuil_getMinScale(GeneralizedHoughGuilPtr ptr);

extern "C" imgproc_API
void GeneralizedHoughGuil_setMaxScale(GeneralizedHoughGuilPtr ptr, double MaxScale);

extern "C" imgproc_API
double GeneralizedHoughGuil_getMaxScale(GeneralizedHoughGuilPtr ptr);

extern "C" imgproc_API
void GeneralizedHoughGuil_setScaleStep(GeneralizedHoughGuilPtr ptr, double ScaleStep);

extern "C" imgproc_API
double GeneralizedHoughGuil_getScaleStep(GeneralizedHoughGuilPtr ptr);

extern "C" imgproc_API
void GeneralizedHoughGuil_setScaleThresh(GeneralizedHoughGuilPtr ptr, int ScaleThresh);

extern "C" imgproc_API
int GeneralizedHoughGuil_getScaleThresh(GeneralizedHoughGuilPtr ptr);

extern "C" imgproc_API
void GeneralizedHoughGuil_setPosThresh(GeneralizedHoughGuilPtr ptr, int PosThresh);

extern "C" imgproc_API
int GeneralizedHoughGuil_getPosThresh(GeneralizedHoughGuilPtr ptr);

extern "C" imgproc_API
struct CLAHEPtr CLAHE_ctor();

extern "C" imgproc_API
void CLAHE_setClipLimit(CLAHEPtr ptr, double ClipLimit);

extern "C" imgproc_API
double CLAHE_getClipLimit(CLAHEPtr ptr);

extern "C" imgproc_API
void CLAHE_setTilesGridSize(CLAHEPtr ptr, struct SizeWrapper TilesGridSize);

extern "C" imgproc_API
struct SizeWrapper CLAHE_getTilesGridSize(CLAHEPtr ptr);

extern "C" imgproc_API
void CLAHE_collectGarbage(CLAHEPtr ptr);

extern "C" imgproc_API
struct LineSegmentDetectorPtr LineSegmentDetector_ctor(
        int refine, double scale, double sigma_scale, double quant,
        double ang_th, double log_eps, double density_th, int n_bins);

extern "C" imgproc_API
struct TensorArray LineSegmentDetector_detect(
        struct LineSegmentDetectorPtr ptr, struct TensorWrapper image,
        struct TensorWrapper lines, bool width, bool prec, bool nfa);

extern "C" imgproc_API
struct TensorWrapper LineSegmentDetector_drawSegments(
        struct LineSegmentDetectorPtr ptr, struct TensorWrapper image, struct TensorWrapper lines);

extern "C" imgproc_API
int LineSegmentDetector_compareSegments(struct LineSegmentDetectorPtr ptr, struct SizeWrapper size, struct TensorWrapper lines1,
                    struct TensorWrapper lines2, struct TensorWrapper image);

extern "C" imgproc_API
struct Subdiv2DPtr Subdiv2D_ctor_default();

extern "C" imgproc_API
struct Subdiv2DPtr Subdiv2D_ctor(struct RectWrapper rect);

extern "C" imgproc_API
void Subdiv2D_dtor(struct Subdiv2DPtr ptr);

extern "C" imgproc_API
void Subdiv2D_initDelaunay(struct Subdiv2DPtr ptr, struct RectWrapper rect);

extern "C" imgproc_API
int Subdiv2D_insert(struct Subdiv2DPtr ptr, struct Point2fWrapper pt);

extern "C" imgproc_API
void Subdiv2D_insert_vector(struct Subdiv2DPtr ptr, struct TensorWrapper ptvec);

extern "C" imgproc_API
struct Vec3iWrapper Subdiv2D_locate(struct Subdiv2DPtr ptr, struct Point2fWrapper pt);

extern "C" imgproc_API
struct Point2fPlusInt Subdiv2D_findNearest(struct Subdiv2DPtr ptr, struct Point2fWrapper pt);

extern "C" imgproc_API
struct TensorWrapper Subdiv2D_getEdgeList(struct Subdiv2DPtr ptr);

extern "C" imgproc_API
struct TensorWrapper Subdiv2D_getTriangleList(struct Subdiv2DPtr ptr);

extern "C" imgproc_API
struct TensorArray Subdiv2D_getVoronoiFacetList(struct Subdiv2DPtr ptr, struct TensorWrapper idx);

extern "C" imgproc_API
struct Point2fPlusInt Subdiv2D_getVertex(struct Subdiv2DPtr ptr, int vertex);

extern "C" imgproc_API
int Subdiv2D_getEdge(struct Subdiv2DPtr ptr, int edge, int nextEdgeType);

extern "C" imgproc_API
int Subdiv2D_nextEdge(struct Subdiv2DPtr ptr, int edge);

extern "C" imgproc_API
int Subdiv2D_rotateEdge(struct Subdiv2DPtr ptr, int edge, int rotate);

extern "C" imgproc_API
int Subdiv2D_symEdge(struct Subdiv2DPtr ptr, int edge);

extern "C" imgproc_API
struct Point2fPlusInt Subdiv2D_edgeOrg(struct Subdiv2DPtr ptr, int edge);

extern "C" imgproc_API
struct Point2fPlusInt Subdiv2D_edgeDst(struct Subdiv2DPtr ptr, int edge);

extern "C" imgproc_API
struct LineIteratorPtr LineIterator_ctor(
        struct TensorWrapper img, struct PointWrapper pt1, struct PointWrapper pt2,
        int connectivity, bool leftToRight);

extern "C" imgproc_API
void LineIterator_dtor(struct LineIteratorPtr ptr);

extern "C" imgproc_API
int LineIterator_count(struct LineIteratorPtr ptr);

extern "C" imgproc_API
struct PointWrapper LineIterator_pos(struct LineIteratorPtr ptr);

extern "C" imgproc_API
void LineIterator_incr(struct LineIteratorPtr ptr);

extern "C" imgproc_API
struct TensorWrapper addWeighted(
        struct TensorWrapper src1, double alpha, struct TensorWrapper src2, double beta,
        double gamma, struct TensorWrapper dst, int dtype);