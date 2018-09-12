#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/stitching.hpp>
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/warpers.hpp"

// stitching_EXPORTS is defined by CMake(add_library)
#if defined (_WIN32) && defined (_MSC_VER)
  #if defined(stitching_EXPORTS)
    #define  stitching_API __declspec(dllexport)
  #else
    #define  stitching_API __declspec(dllimport)
  #endif /* stitching_EXPORTS */
#else /* defined (_WIN32) */
 #define stitching_API
#endif

namespace detail = cv::detail;

struct ClassArray {
	void *data;
	int size;

	ClassArray() {}

	ClassArray(const std::vector<cv::detail::ImageFeatures> & vec);
	ClassArray(const std::vector<cv::detail::MatchesInfo> & vec);
	ClassArray(const std::vector<cv::detail::CameraParams> & vec);

	operator std::vector<cv::detail::ImageFeatures>();
	operator std::vector<cv::detail::MatchesInfo>();
	operator std::vector<cv::detail::CameraParams>();
};

extern "C" stitching_API
struct RectPlusBool detail_overlapRoi(
	struct PointWrapper tl1, struct PointWrapper tl2,
	struct SizeWrapper sz1, struct SizeWrapper sz2);

extern "C" stitching_API
struct RectWrapper detail_resultRoi(
	struct PointArray corners,
	struct SizeArray sizes);

extern "C" stitching_API
struct RectWrapper detail_resultRoi2(
		struct PointArray corners,
		struct TensorArray images);

extern "C" stitching_API
struct RectWrapper detail_resultRoiIntersection(
	struct PointArray corners,
	struct SizeArray sizes);

extern "C" stitching_API
struct PointWrapper detail_resultTl(
	struct PointArray corners);

extern "C" stitching_API
struct TensorWrapper detail_selectRandomSubset(
	int count, int size);

extern "C" stitching_API
int detail_stitchingLogLevel();

//CameraParams 

extern "C"
struct CameraParamsPtr {
    void *ptr;

	CameraParamsPtr() {}

    inline detail::CameraParams * operator->() {
			return static_cast<detail::CameraParams *>(ptr); }
    inline CameraParamsPtr(detail::CameraParams *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct CameraParamsPtr CameraParams_ctor();

extern "C" stitching_API
struct CameraParamsPtr CameraParams_ctor2(
	struct CameraParamsPtr other);

extern "C" stitching_API
void CameraParams_dtor(
	struct CameraParamsPtr ptr);

struct TensorWrapper CameraParams_K(
	struct CameraParamsPtr ptr);

//DisjointSets

extern "C"
struct DisjointSetsPtr {
    void *ptr;

    inline detail::DisjointSets * operator->() {
			return static_cast<detail::DisjointSets *>(ptr); }
    inline DisjointSetsPtr(detail::DisjointSets *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct DisjointSetsPtr DisjointSets_ctor(
	int elem_count);

extern "C" stitching_API
void DisjointSets_dtor(
	struct DisjointSetsPtr ptr);

extern "C" stitching_API
void DisjointSets_createOneElemSets(
	struct DisjointSetsPtr ptr, int elem_count);

extern "C" stitching_API
int DisjointSets_findSetByElem(
	struct DisjointSetsPtr ptr, int elem);

extern "C" stitching_API
int DisjointSets_mergeSets(
	struct DisjointSetsPtr ptr,
	int set1, int set2);

//Graph

extern "C"
struct GraphPtr {
    void *ptr;

    inline detail::Graph * operator->() {
			return static_cast<detail::Graph *>(ptr); }
	GraphPtr() {}
    inline GraphPtr(detail::Graph *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct GraphPtr Graph_ctor(
	int num_vertices);

extern "C" stitching_API
void Graph_dtor(
	struct GraphPtr ptr);

extern "C" stitching_API
void Graph_addEdge(
	struct GraphPtr ptr, int from, int to, float weight);

extern "C" stitching_API
void Graph_create(
	struct GraphPtr ptr, int num_vertices);

//TODO add template<typename B> B forEach(B body) const

extern "C" stitching_API
int Graph_numVertices(
	struct GraphPtr ptr);

//TODO add template<typename B> B walkBreadthFirst (int from, B body) const

//GraphEdge

extern "C"
struct GraphEdgePtr {
    void *ptr;

    inline detail::GraphEdge * operator->() {
			return static_cast<detail::GraphEdge *>(ptr); }
    inline GraphEdgePtr(detail::GraphEdge *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct GraphEdgePtr GraphEdge_ctor(
	int from, int to, float weight);

extern "C" stitching_API
void GraphEdge_dtor(
	struct GraphEdgePtr ptr);

//Timelapser

extern "C"
struct TimelapserPtr {
    void *ptr;

    inline detail::Timelapser * operator->() {
			return static_cast<detail::Timelapser *>(ptr); }
    inline TimelapserPtr(detail::Timelapser * ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct TimelapserPtr Timelapser_ctor(
	int type);

extern "C" stitching_API
void Timelapser_dtor(
	struct TimelapserPtr ptr);

extern "C" stitching_API
struct TensorWrapper Timelapser_getDst(
		struct TimelapserPtr ptr);

extern "C" stitching_API
void Timelapser_initialize(
	struct TimelapserPtr ptr, struct PointArray corners,
	struct SizeArray sizes);

extern "C" stitching_API
void Timelapser_process(
	struct TimelapserPtr ptr, struct TensorWrapper img,
	struct TensorWrapper mask, struct PointWrapper tl);

//TimelapserCrop

extern "C"
struct TimelapserCropPtr {
    void *ptr;

    inline detail::TimelapserCrop * operator->() {
			return static_cast<detail::TimelapserCrop *>(ptr); }
    inline TimelapserCropPtr(detail::TimelapserCrop *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
void TimelapserCrop_initialize(
	struct TimelapserCropPtr ptr, struct PointArray corners,
	struct SizeArray sizes); 


//*************Features Finding and Images Matching**************


//MatchesInfo

extern "C"
struct MatchesInfoPtr {
	void *ptr;

	MatchesInfoPtr() {};

	inline detail::MatchesInfo * operator->() {
		return static_cast<detail::MatchesInfo *>(ptr); }
	inline MatchesInfoPtr(detail::MatchesInfo *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct MatchesInfoPtr MatchesInfo_ctor();

extern "C" stitching_API
struct MatchesInfoPtr MatchesInfo_ctor2(
		struct MatchesInfoPtr other);

extern "C" stitching_API
void MatchesInfo_dtor(
		struct MatchesInfoPtr ptr);


//ImageFeatures

extern "C"
struct ImageFeaturesPtr {
	void *ptr;

	ImageFeaturesPtr() {}

	inline detail::ImageFeatures * operator->() {
		return static_cast<detail::ImageFeatures *>(ptr); }
	inline ImageFeaturesPtr(detail::ImageFeatures *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct ImageFeaturesPtr ImageFeatures_ctor();

extern "C" stitching_API
void ImageFeatures_dtor(
		struct ImageFeaturesPtr ptr);

//FeaturesFinder

extern "C"
struct FeaturesFinderPtr {
    void *ptr;

    inline detail::FeaturesFinder * operator->() {
			return static_cast<detail::FeaturesFinder *>(ptr); }
    inline FeaturesFinderPtr(detail::FeaturesFinder *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
void FeaturesFinder_dtor(
	struct FeaturesFinderPtr ptr);

extern "C" stitching_API
void FeaturesFinder_collectGarbage(
	struct FeaturesFinderPtr ptr);

extern "C" stitching_API
struct ImageFeaturesPtr FeaturesFinder_call(
		struct FeaturesFinderPtr ptr, struct TensorWrapper image);

extern "C" stitching_API
struct ImageFeaturesPtr FeaturesFinder_call2(
		struct FeaturesFinderPtr ptr, struct TensorWrapper image,
		struct RectArray);

//OrbFeaturesFinder

extern "C"
struct OrbFeaturesFinderPtr {
	void *ptr;

	inline detail::OrbFeaturesFinder * operator->() {
		return static_cast<detail::OrbFeaturesFinder *>(ptr); }
	inline OrbFeaturesFinderPtr(detail::OrbFeaturesFinder *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct OrbFeaturesFinderPtr OrbFeaturesFinder_ctor(
		struct SizeWrapper _grid_size, int nfeatures, float scaleFactor, int nlevels);

//SurfFeaturesFinder

extern "C"
struct SurfFeaturesFinderPtr {
	void *ptr;

	inline detail::SurfFeaturesFinder * operator->() {
		return static_cast<detail::SurfFeaturesFinder *>(ptr); }
	inline SurfFeaturesFinderPtr(detail::SurfFeaturesFinder *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct SurfFeaturesFinderPtr SurfFeaturesFinder_ctor(
		double hess_thresh, int num_octaves, int num_layers,
		int num_octaves_descr, int num_layers_descr);

//FeaturesMatcher

extern "C"
struct FeaturesMatcherPtr {
	void *ptr;

	inline detail::FeaturesMatcher * operator->() {
		return static_cast<detail::FeaturesMatcher *>(ptr); }
	inline FeaturesMatcherPtr(detail::FeaturesMatcher *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
void FeaturesMatcher_dtor(
		struct FeaturesMatcherPtr ptr);

extern "C" stitching_API
void FeaturesMatcher_collectGarbage(
		struct FeaturesMatcherPtr ptr);

extern "C" stitching_API
bool FeaturesMatcher_isThreadSafe(
		struct FeaturesMatcherPtr ptr);

extern "C" stitching_API
struct MatchesInfoPtr FeaturesMatcher_call(
		struct FeaturesMatcherPtr ptr, struct ImageFeaturesPtr features1,
		struct ImageFeaturesPtr features2);

//BestOf2NearestMatcher

extern "C"
struct BestOf2NearestMatcherPtr {
	void *ptr;

	inline detail::BestOf2NearestMatcher * operator->() {
		return static_cast<detail::BestOf2NearestMatcher *>(ptr); }
	inline BestOf2NearestMatcherPtr(detail::BestOf2NearestMatcher *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct BestOf2NearestMatcherPtr BestOf2NearestMatcher_ctor(
	bool try_use_gpu, float match_conf,
	int num_matches_thresh1, int num_matches_thresh2);

extern "C" stitching_API
void BestOf2NearestMatcher_collectGarbage(
	struct BestOf2NearestMatcherPtr ptr);

//BestOf2NearestRangeMatcher

extern "C"
struct BestOf2NearestRangeMatcherPtr {
	void *ptr;

	inline detail::BestOf2NearestRangeMatcher * operator->() {
		return static_cast<detail::BestOf2NearestRangeMatcher *>(ptr); }
	inline BestOf2NearestRangeMatcherPtr(detail::BestOf2NearestRangeMatcher *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct BestOf2NearestRangeMatcherPtr BestOf2NearestRangeMatcher_ctor(
		int range_width, bool try_use_gpu, float match_conf,
		int num_matches_thresh1, int num_matches_thresh2);

extern "C" stitching_API
void BestOf2NearestRangeMatcher_call(
		struct BestOf2NearestRangeMatcherPtr ptr, struct ClassArray features,
		struct ClassArray pairwise_matches, struct TensorWrapper mask);


//**********************Rotation Estimation********************************

struct GraphPtrPlusTensor {
	struct GraphPtr graph;
	struct TensorWrapper tensor;
};

struct BoolPlusClassArray {
	bool val;
	struct ClassArray array;
};

extern "C" stitching_API
struct GraphPtrPlusTensor detail_findMaxSpanningTree(
		int num_images, struct ClassArray pairwise_matches);

extern "C" stitching_API
struct TensorWrapper detail_leaveBiggestComponent(
		struct ClassArray features, struct ClassArray pairwise_matches, float conf_threshold);

extern "C" stitching_API
struct StringArray detail_matchesGraphAsString(
		struct StringArray pathes, struct ClassArray pairwise_matches, float conf_threshold);

extern "C" stitching_API
void detail_waveCorrect(
		struct TensorArray rmats, int kind);

//Estimator

extern "C"
struct EstimatorPtr {
	void *ptr;

	inline detail::Estimator * operator->() {
		return static_cast<detail::Estimator *>(ptr); }
	inline EstimatorPtr(detail::Estimator *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
void Estimator_dtor(
		struct EstimatorPtr ptr);

extern "C" stitching_API
struct BoolPlusClassArray Estimator_call(
		struct EstimatorPtr ptr, struct ClassArray features, struct ClassArray 	pairwise_matches);

//HomographyBasedEstimator

extern "C"
struct HomographyBasedEstimatorPtr {
	void *ptr;

	inline detail::HomographyBasedEstimator * operator->() {
		return static_cast<detail::HomographyBasedEstimator *>(ptr); }
	inline HomographyBasedEstimatorPtr(detail::HomographyBasedEstimator *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct HomographyBasedEstimatorPtr HomographyBasedEstimator_ctor(
		bool is_focals_estimated);

//BundleAdjusterBase

extern "C"
struct BundleAdjusterBasePtr {
	void *ptr;

	inline detail::BundleAdjusterBase * operator->() {
		return static_cast<detail::BundleAdjusterBase *>(ptr); }
	inline BundleAdjusterBasePtr(detail::BundleAdjusterBase *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
double BundleAdjusterBase_confThresh(
		struct BundleAdjusterBasePtr ptr);

extern "C" stitching_API
struct TensorWrapper BundleAdjusterBase_refinementMask(
		struct BundleAdjusterBasePtr ptr);


extern "C" stitching_API
void BundleAdjusterBase_setConfThresh(
		struct BundleAdjusterBasePtr ptr, double conf_thresh);


extern "C" stitching_API
void BundleAdjusterBase_setRefinementMask(
		struct BundleAdjusterBasePtr ptr, struct TensorWrapper mask);

extern "C" stitching_API
void BundleAdjusterBase_setTermCriteria(
		struct BundleAdjusterBasePtr ptr, struct TermCriteriaWrapper term_criteria);

extern "C" stitching_API
struct TermCriteriaWrapper BundleAdjusterBase_termCriteria(
		struct BundleAdjusterBasePtr ptr);

//BundleAdjusterRay

extern "C"
struct BundleAdjusterRayPtr {
	void *ptr;

	inline detail::BundleAdjusterRay * operator->() {
		return static_cast<detail::BundleAdjusterRay *>(ptr); }
	inline BundleAdjusterRayPtr(detail::BundleAdjusterRay *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct BundleAdjusterRayPtr BundleAdjusterRay_ctor();

//BundleAdjusterReproj

extern "C"
struct BundleAdjusterReprojPtr {
	void *ptr;

	inline detail::BundleAdjusterReproj * operator->() {
		return static_cast<detail::BundleAdjusterReproj *>(ptr); }
	inline BundleAdjusterReprojPtr(detail::BundleAdjusterReproj *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct BundleAdjusterReprojPtr BundleAdjusterReproj_ctor();


//************************Autocalibration********************

struct focalsFromHomographyRetval {
	double f0, f1;
	bool f0_ok, f1_ok;
};

extern "C" stitching_API
struct TensorPlusBool detail_calibrateRotatingCamera(
		struct TensorArray Hs);

extern "C" stitching_API
struct TensorWrapper detail_estimateFocal(
		struct ClassArray features, struct ClassArray pairwise_matches);

extern "C" stitching_API
struct focalsFromHomographyRetval detail_focalsFromHomography(
		struct TensorWrapper H);


//***********************Images Warping***********************

//ProjectorBase

extern "C"
struct ProjectorBasePtr {
	void *ptr;

	inline detail::ProjectorBase * operator->() {
		return static_cast<detail::ProjectorBase *>(ptr); }
	inline ProjectorBasePtr(detail::ProjectorBase *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct ProjectorBasePtr ProjectorBase_ctor();

extern "C" stitching_API
void ProjectorBase_dtor(
		struct ProjectorBasePtr ptr);

extern "C" stitching_API
void ProjectorBase_setCameraParams(
		struct ProjectorBasePtr ptr, struct TensorWrapper K,
		struct TensorWrapper R, struct TensorWrapper T);

//CompressedRectilinearPortraitProjector

extern "C"
struct CompressedRectilinearPortraitProjectorPtr {
	void *ptr;

	inline detail::CompressedRectilinearPortraitProjector * operator->() {
		return static_cast<detail::CompressedRectilinearPortraitProjector *>(ptr); }
	inline CompressedRectilinearPortraitProjectorPtr(detail::CompressedRectilinearPortraitProjector *ptr) {
		this->ptr = ptr; }
};

extern "C" stitching_API
struct CompressedRectilinearPortraitProjectorPtr CompressedRectilinearPortraitProjector_ctor();

extern "C" stitching_API
void CompressedRectilinearPortraitProjector_dtor(
		struct CompressedRectilinearPortraitProjectorPtr ptr);

extern "C" stitching_API
struct FloatArray CompressedRectilinearPortraitProjector_mapBackward(
		struct CompressedRectilinearPortraitProjectorPtr ptr, float u, float v);

extern "C" stitching_API
struct FloatArray CompressedRectilinearPortraitProjector_mapForward(
		struct CompressedRectilinearPortraitProjectorPtr ptr, float x, float y);

//CompressedRectilinearProjector

extern "C"
struct CompressedRectilinearProjectorPtr {
	void *ptr;

	inline detail::CompressedRectilinearProjector * operator->() {
		return static_cast<detail::CompressedRectilinearProjector  *>(ptr); }
	inline CompressedRectilinearProjectorPtr(detail::CompressedRectilinearProjector *ptr) {
		this->ptr = ptr; }
};

extern "C" stitching_API
struct CompressedRectilinearProjectorPtr CompressedRectilinearProjector_ctor();

extern "C" stitching_API
void CompressedRectilinearProjector_dtor(
		struct CompressedRectilinearProjectorPtr ptr);

extern "C" stitching_API
struct FloatArray CompressedRectilinearProjector_mapBackward(
		struct CompressedRectilinearProjectorPtr ptr, float u, float v);

extern "C" stitching_API
struct FloatArray CompressedRectilinearProjector_mapForward(
		struct CompressedRectilinearProjectorPtr ptr, float x, float y);

//CylindricalPortraitProjector

extern "C"
struct CylindricalPortraitProjectorPtr {
	void *ptr;

	inline detail::CylindricalPortraitProjector * operator->() {
		return static_cast<detail::CylindricalPortraitProjector  *>(ptr); }
	inline CylindricalPortraitProjectorPtr(detail::CylindricalPortraitProjector *ptr) {
		this->ptr = ptr; }
};

extern "C" stitching_API
struct CylindricalPortraitProjectorPtr CylindricalPortraitProjector_ctor();

extern "C" stitching_API
void CylindricalPortraitProjector_dtor(
		struct CylindricalPortraitProjectorPtr ptr);

extern "C" stitching_API
struct FloatArray CylindricalPortraitProjector_mapBackward(
		struct CylindricalPortraitProjectorPtr ptr, float u, float v);

extern "C" stitching_API
struct FloatArray CylindricalPortraitProjector_mapForward(
		struct CylindricalPortraitProjectorPtr ptr, float x, float y);

//CylindricalProjector

extern "C"
struct CylindricalProjectorPtr {
	void *ptr;

	inline detail::CylindricalProjector * operator->() {
		return static_cast<detail::CylindricalProjector  *>(ptr); }
	inline CylindricalProjectorPtr(detail::CylindricalProjector *ptr) {
		this->ptr = ptr; }
};

extern "C" stitching_API
struct CylindricalProjectorPtr CylindricalProjector_ctor();

extern "C" stitching_API
void CylindricalProjector_dtor(
		struct CylindricalProjectorPtr ptr);

extern "C" stitching_API
struct FloatArray CylindricalProjector_mapBackward(
		struct CylindricalProjectorPtr ptr, float u, float v);

extern "C" stitching_API
struct FloatArray CylindricalProjector_mapForward(
		struct CylindricalProjectorPtr ptr, float x, float y);

//FisheyeProjector

extern "C"
struct FisheyeProjectorPtr {
	void *ptr;

	inline detail::FisheyeProjector * operator->() {
		return static_cast<detail::FisheyeProjector  *>(ptr); }
	inline FisheyeProjectorPtr(detail::FisheyeProjector *ptr) {
		this->ptr = ptr; }
};

extern "C" stitching_API
struct FisheyeProjectorPtr FisheyeProjector_ctor();

extern "C" stitching_API
void FisheyeProjector_dtor(
		struct FisheyeProjectorPtr ptr);

extern "C" stitching_API
struct FloatArray FisheyeProjector_mapBackward(
		struct FisheyeProjectorPtr ptr, float u, float v);

extern "C" stitching_API
struct FloatArray FisheyeProjector_mapForward(
		struct FisheyeProjectorPtr ptr, float x, float y);

//MercatorProjector

extern "C"
struct MercatorProjectorPtr {
	void *ptr;

	inline detail::MercatorProjector * operator->() {
		return static_cast<detail::MercatorProjector  *>(ptr); }
	inline MercatorProjectorPtr(detail::MercatorProjector *ptr) {
		this->ptr = ptr; }
};

extern "C" stitching_API
struct MercatorProjectorPtr MercatorProjector_ctor();

extern "C" stitching_API
void MercatorProjector_dtor(
		struct MercatorProjectorPtr ptr);

extern "C" stitching_API
struct FloatArray MercatorProjector_mapBackward(
		struct MercatorProjectorPtr ptr, float u, float v);

extern "C" stitching_API
struct FloatArray MercatorProjector_mapForward(
		struct MercatorProjectorPtr ptr, float x, float y);

//PaniniPortraitProjector

extern "C"
struct PaniniPortraitProjectorPtr {
	void *ptr;

	inline detail::PaniniPortraitProjector * operator->() {
		return static_cast<detail::PaniniPortraitProjector  *>(ptr); }
	inline PaniniPortraitProjectorPtr(detail::PaniniPortraitProjector *ptr) {
		this->ptr = ptr; }
};

extern "C" stitching_API
struct PaniniPortraitProjectorPtr PaniniPortraitProjector_ctor();

extern "C" stitching_API
void PaniniPortraitProjector_dtor(
		struct PaniniPortraitProjectorPtr ptr);

extern "C" stitching_API
struct FloatArray PaniniPortraitProjector_mapBackward(
		struct PaniniPortraitProjectorPtr ptr, float u, float v);

extern "C" stitching_API
struct FloatArray PaniniPortraitProjector_mapForward(
		struct PaniniPortraitProjectorPtr ptr, float x, float y);

//PaniniProjector

extern "C"
struct PaniniProjectorPtr {
	void *ptr;

	inline detail::PaniniProjector * operator->() {
		return static_cast<detail::PaniniProjector  *>(ptr); }
	inline PaniniProjectorPtr(detail::PaniniProjector *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct PaniniProjectorPtr PaniniProjector_ctor();

extern "C" stitching_API
void PaniniProjector_dtor(
		struct PaniniProjectorPtr ptr);

extern "C" stitching_API
struct FloatArray PaniniProjector_mapBackward(
		struct PaniniProjectorPtr ptr, float u, float v);

extern "C" stitching_API
struct FloatArray PaniniProjector_mapForward(
		struct PaniniProjectorPtr ptr, float x, float y);

//PlanePortraitProjector

extern "C"
struct PlanePortraitProjectorPtr {
	void *ptr;

	inline detail::PlanePortraitProjector * operator->() {
		return static_cast<detail::PlanePortraitProjector  *>(ptr); }
	inline PlanePortraitProjectorPtr(detail::PlanePortraitProjector *ptr) {
		this->ptr = ptr; }
};

extern "C" stitching_API
struct PlanePortraitProjectorPtr PlanePortraitProjector_ctor();

extern "C" stitching_API
void PlanePortraitProjector_dtor(
		struct PlanePortraitProjectorPtr ptr);

extern "C" stitching_API
struct FloatArray PlanePortraitProjector_mapBackward(
		struct PlanePortraitProjectorPtr ptr, float u, float v);

extern "C" stitching_API
struct FloatArray PlanePortraitProjector_mapForward(
		struct PlanePortraitProjectorPtr ptr, float x, float y);

//PlaneProjector

extern "C"
struct PlaneProjectorPtr {
	void *ptr;

	inline detail::PlaneProjector * operator->() {
		return static_cast<detail::PlaneProjector  *>(ptr); }
	inline PlaneProjectorPtr(detail::PlaneProjector *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct PlaneProjectorPtr PlaneProjector_ctor();

extern "C" stitching_API
void PlaneProjector_dtor(
		struct PlaneProjectorPtr ptr);

extern "C" stitching_API
struct FloatArray PlaneProjector_mapBackward(
		struct PlaneProjectorPtr ptr, float u, float v);

extern "C" stitching_API
struct FloatArray PlaneProjector_mapForward(
		struct PlaneProjectorPtr ptr, float x, float y);

//SphericalPortraitProjector

extern "C"
struct SphericalPortraitProjectorPtr {
	void *ptr;

	inline detail::SphericalPortraitProjector * operator->() {
		return static_cast<detail::SphericalPortraitProjector  *>(ptr); }
	inline SphericalPortraitProjectorPtr(detail::SphericalPortraitProjector *ptr) {
		this->ptr = ptr; }
};

extern "C" stitching_API
struct SphericalPortraitProjectorPtr SphericalPortraitProjector_ctor();

extern "C" stitching_API
void SphericalPortraitProjector_dtor(
		struct SphericalPortraitProjectorPtr ptr);

extern "C" stitching_API
struct FloatArray SphericalPortraitProjector_mapBackward(
		struct SphericalPortraitProjectorPtr ptr, float u, float v);

extern "C" stitching_API
struct FloatArray SphericalPortraitProjector_mapForward(
		struct SphericalPortraitProjectorPtr ptr, float x, float y);

//SphericalProjector

extern "C"
struct SphericalProjectorPtr {
	void *ptr;

	inline detail::SphericalProjector * operator->() {
		return static_cast<detail::SphericalProjector  *>(ptr); }
	inline SphericalProjectorPtr(detail::SphericalProjector *ptr) {
		this->ptr = ptr; }
};

extern "C" stitching_API
struct SphericalProjectorPtr SphericalProjector_ctor();

extern "C" stitching_API
void SphericalProjector_dtor(
		struct SphericalProjectorPtr ptr);

extern "C" stitching_API
struct FloatArray SphericalProjector_mapBackward(
		struct SphericalProjectorPtr ptr, float u, float v);

extern "C" stitching_API
struct FloatArray SphericalProjector_mapForward(
		struct SphericalProjectorPtr ptr, float x, float y);

//StereographicProjector

extern "C"
struct StereographicProjectorPtr {
	void *ptr;

	inline detail::StereographicProjector * operator->() {
		return static_cast<detail::StereographicProjector  *>(ptr); }
	inline StereographicProjectorPtr(detail::StereographicProjector *ptr) {
		this->ptr = ptr; }
};

extern "C" stitching_API
struct StereographicProjectorPtr StereographicProjector_ctor();

extern "C" stitching_API
void StereographicProjector_dtor(
		struct StereographicProjectorPtr ptr);

extern "C" stitching_API
struct FloatArray StereographicProjector_mapBackward(
		struct StereographicProjectorPtr ptr, float u, float v);

extern "C" stitching_API
struct FloatArray StereographicProjector_mapForward(
		struct StereographicProjectorPtr ptr, float x, float y);

//TransverseMercatorProjector

extern "C"
struct TransverseMercatorProjectorPtr {
	void *ptr;

	inline detail::TransverseMercatorProjector * operator->() {
		return static_cast<detail::TransverseMercatorProjector  *>(ptr); }
	inline TransverseMercatorProjectorPtr(detail::TransverseMercatorProjector *ptr) {
		this->ptr = ptr; }
};

extern "C" stitching_API
struct TransverseMercatorProjectorPtr TransverseMercatorProjector_ctor();

extern "C" stitching_API
void TransverseMercatorProjector_dtor(
		struct TransverseMercatorProjectorPtr ptr);

extern "C" stitching_API
struct FloatArray TransverseMercatorProjector_mapBackward(
		struct TransverseMercatorProjectorPtr ptr, float u, float v);

extern "C" stitching_API
struct FloatArray TransverseMercatorProjector_mapForward(
		struct TransverseMercatorProjectorPtr ptr, float x, float y);

//RotationWarper

extern "C"
struct RotationWarperPtr {
	void *ptr;

	inline detail::RotationWarper * operator->() {
		return static_cast<detail::RotationWarper  *>(ptr); }
	inline RotationWarperPtr(detail::RotationWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
void RotationWarper_dtor(
		struct RotationWarperPtr ptr);

extern "C" stitching_API
struct TensorArrayPlusRect RotationWarper_buildMaps(
		struct RotationWarperPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" stitching_API
float RotationWarper_getScale(
		struct RotationWarperPtr ptr);

extern "C" stitching_API
void RotationWarper_setScale(
		struct RotationWarperPtr ptr, float val);

extern "C" stitching_API
struct TensorPlusPoint RotationWarper_warp(
		struct RotationWarperPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

extern "C" stitching_API
struct TensorWrapper RotationWarper_warpBackward(
		struct RotationWarperPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

extern "C" stitching_API
struct Point2fWrapper RotationWarper_warpPoint(
		struct RotationWarperPtr ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

extern "C" stitching_API
struct RectWrapper RotationWarper_warpRoi(
		struct RotationWarperPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

//RotationWarperBase_CompressedRectilinearPortraitProjector

extern "C"
struct RotationWarperBase_CompressedRectilinearPortraitProjectorPtr {
	void *ptr;

	inline detail::RotationWarperBase<detail::CompressedRectilinearPortraitProjector> * operator->() {
		return static_cast<detail::RotationWarperBase<detail::CompressedRectilinearPortraitProjector>  *>(ptr);
	}
	inline RotationWarperBase_CompressedRectilinearPortraitProjectorPtr(
			detail::RotationWarperBase<detail::CompressedRectilinearPortraitProjector> *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct RotationWarperBase_CompressedRectilinearPortraitProjectorPtr
				RotationWarperBase_CompressedRectilinearPortraitProjector_ctor();

extern "C" stitching_API
struct TensorArrayPlusRect RotationWarperBase_CompressedRectilinearPortraitProjector_buildMaps(
		struct RotationWarperBase_CompressedRectilinearPortraitProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" stitching_API
float RotationWarperBase_CompressedRectilinearPortraitProjector_getScale(
		struct RotationWarperBase_CompressedRectilinearPortraitProjectorPtr ptr);

extern "C" stitching_API
void RotationWarperBase_CompressedRectilinearPortraitProjector_setScale(
		struct RotationWarperBase_CompressedRectilinearPortraitProjectorPtr ptr, float val);

extern "C" stitching_API
struct TensorPlusPoint RotationWarperBase_CompressedRectilinearPortraitProjector_warp(
		struct RotationWarperBase_CompressedRectilinearPortraitProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

extern "C" stitching_API
struct TensorWrapper RotationWarperBase_CompressedRectilinearPortraitProjector_warpBackward(
		struct RotationWarperBase_CompressedRectilinearPortraitProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

extern "C" stitching_API
struct Point2fWrapper RotationWarperBase_CompressedRectilinearPortraitProjector_warpPoint(
		struct RotationWarperBase_CompressedRectilinearPortraitProjectorPtr ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

extern "C" stitching_API
struct RectWrapper RotationWarperBase_CompressedRectilinearPortraitProjector_warpRoi(
		struct RotationWarperBase_CompressedRectilinearPortraitProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

//RotationWarperBase_CompressedRectilinearProjector

extern "C"
struct RotationWarperBase_CompressedRectilinearProjectorPtr {
	void *ptr;

	inline detail::RotationWarperBase<detail::CompressedRectilinearProjector> * operator->() {
		return static_cast<detail::RotationWarperBase<detail::CompressedRectilinearProjector>  *>(ptr);
	}
	inline RotationWarperBase_CompressedRectilinearProjectorPtr(
			detail::RotationWarperBase<detail::CompressedRectilinearProjector> *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct RotationWarperBase_CompressedRectilinearProjectorPtr
			RotationWarperBase_CompressedRectilinearProjector_ctor();

extern "C" stitching_API
struct TensorArrayPlusRect RotationWarperBase_CompressedRectilinearProjector_buildMaps(
		struct RotationWarperBase_CompressedRectilinearProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" stitching_API
float RotationWarperBase_CompressedRectilinearProjector_getScale(
		struct RotationWarperBase_CompressedRectilinearProjectorPtr ptr);

extern "C" stitching_API
void RotationWarperBase_CompressedRectilinearProjector_setScale(
		struct RotationWarperBase_CompressedRectilinearProjectorPtr ptr, float val);

extern "C" stitching_API
struct TensorPlusPoint RotationWarperBase_CompressedRectilinearProjector_warp(
		struct RotationWarperBase_CompressedRectilinearProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

extern "C" stitching_API
struct TensorWrapper RotationWarperBase_CompressedRectilinearProjector_warpBackward(
		struct RotationWarperBase_CompressedRectilinearProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

extern "C" stitching_API
struct Point2fWrapper RotationWarperBase_CompressedRectilinearProjector_warpPoint(
		struct RotationWarperBase_CompressedRectilinearProjectorPtr ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

extern "C" stitching_API
struct RectWrapper RotationWarperBase_CompressedRectilinearProjector_warpRoi(
		struct RotationWarperBase_CompressedRectilinearProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

//RotationWarperBase_CylindricalPortraitProjector

extern "C"
struct RotationWarperBase_CylindricalPortraitProjectorPtr {
	void *ptr;

	inline detail::RotationWarperBase<detail::CylindricalPortraitProjector> * operator->() {
		return static_cast<detail::RotationWarperBase<detail::CylindricalPortraitProjector>  *>(ptr);
	}
	inline RotationWarperBase_CylindricalPortraitProjectorPtr(
			detail::RotationWarperBase<detail::CylindricalPortraitProjector> *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct RotationWarperBase_CylindricalPortraitProjectorPtr
			RotationWarperBase_CylindricalPortraitProjector_ctor();

extern "C" stitching_API
struct TensorArrayPlusRect RotationWarperBase_CylindricalPortraitProjector_buildMaps(
		struct RotationWarperBase_CylindricalPortraitProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" stitching_API
float RotationWarperBase_CylindricalPortraitProjector_getScale(
		struct RotationWarperBase_CylindricalPortraitProjectorPtr ptr);

extern "C" stitching_API
void RotationWarperBase_CylindricalPortraitProjector_setScale(
		struct RotationWarperBase_CylindricalPortraitProjectorPtr ptr, float val);

extern "C" stitching_API
struct TensorPlusPoint RotationWarperBase_CylindricalPortraitProjector_warp(
		struct RotationWarperBase_CylindricalPortraitProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

extern "C" stitching_API
struct TensorWrapper RotationWarperBase_CylindricalPortraitProjector_warpBackward(
		struct RotationWarperBase_CylindricalPortraitProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

extern "C" stitching_API
struct Point2fWrapper RotationWarperBase_CylindricalPortraitProjector_warpPoint(
		struct RotationWarperBase_CylindricalPortraitProjectorPtr ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

extern "C" stitching_API
struct RectWrapper RotationWarperBase_CylindricalPortraitProjector_warpRoi(
		struct RotationWarperBase_CylindricalPortraitProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

//RotationWarperBase_CylindricalProjector

extern "C"
struct RotationWarperBase_CylindricalProjectorPtr {
	void *ptr;

	inline detail::RotationWarperBase<detail::CylindricalProjector> * operator->() {
		return static_cast<detail::RotationWarperBase<detail::CylindricalProjector>  *>(ptr);
	}
	inline RotationWarperBase_CylindricalProjectorPtr(
			detail::RotationWarperBase<detail::CylindricalProjector> *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct RotationWarperBase_CylindricalProjectorPtr RotationWarperBase_CylindricalProjector_ctor();

extern "C" stitching_API
struct TensorArrayPlusRect RotationWarperBase_CylindricalProjector_buildMaps(
		struct RotationWarperBase_CylindricalProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" stitching_API
float RotationWarperBase_CylindricalProjector_getScale(
		struct RotationWarperBase_CylindricalProjectorPtr ptr);

extern "C" stitching_API
void RotationWarperBase_CylindricalProjector_setScale(
		struct RotationWarperBase_CylindricalProjectorPtr ptr, float val);

extern "C" stitching_API
struct TensorPlusPoint RotationWarperBase_CylindricalProjector_warp(
		struct RotationWarperBase_CylindricalProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

extern "C" stitching_API
struct TensorWrapper RotationWarperBase_CylindricalProjector_warpBackward(
		struct RotationWarperBase_CylindricalProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

extern "C" stitching_API
struct Point2fWrapper RotationWarperBase_CylindricalProjector_warpPoint(
		struct RotationWarperBase_CylindricalProjectorPtr ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

extern "C" stitching_API
struct RectWrapper RotationWarperBase_CylindricalProjector_warpRoi(
		struct RotationWarperBase_CylindricalProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

//RotationWarperBase_FisheyeProjector

extern "C"
struct RotationWarperBase_FisheyeProjectorPtr {
	void *ptr;

	inline detail::RotationWarperBase<detail::FisheyeProjector> * operator->() {
		return static_cast<detail::RotationWarperBase<detail::FisheyeProjector>  *>(ptr);
	}
	inline RotationWarperBase_FisheyeProjectorPtr(
			detail::RotationWarperBase<detail::FisheyeProjector> *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct RotationWarperBase_FisheyeProjectorPtr RotationWarperBase_FisheyeProjector_ctor();

extern "C" stitching_API
struct TensorArrayPlusRect RotationWarperBase_FisheyeProjector_buildMaps(
		struct RotationWarperBase_FisheyeProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" stitching_API
float RotationWarperBase_FisheyeProjector_getScale(
		struct RotationWarperBase_FisheyeProjectorPtr ptr);

extern "C" stitching_API
void RotationWarperBase_FisheyeProjector_setScale(
		struct RotationWarperBase_FisheyeProjectorPtr ptr, float val);

extern "C" stitching_API
struct TensorPlusPoint RotationWarperBase_FisheyeProjector_warp(
		struct RotationWarperBase_FisheyeProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

extern "C" stitching_API
struct TensorWrapper RotationWarperBase_FisheyeProjector_warpBackward(
		struct RotationWarperBase_FisheyeProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

extern "C" stitching_API
struct Point2fWrapper RotationWarperBase_FisheyeProjector_warpPoint(
		struct RotationWarperBase_FisheyeProjectorPtr ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

extern "C" stitching_API
struct RectWrapper RotationWarperBase_FisheyeProjector_warpRoi(
		struct RotationWarperBase_FisheyeProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

//RotationWarperBase_MercatorProjector

extern "C"
struct RotationWarperBase_MercatorProjectorPtr {
	void *ptr;

	inline detail::RotationWarperBase<detail::MercatorProjector> * operator->() {
		return static_cast<detail::RotationWarperBase<detail::MercatorProjector>  *>(ptr);
	}
	inline RotationWarperBase_MercatorProjectorPtr(
			detail::RotationWarperBase<detail::MercatorProjector> *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct RotationWarperBase_MercatorProjectorPtr RotationWarperBase_MercatorProjector_ctor();

extern "C" stitching_API
struct TensorArrayPlusRect RotationWarperBase_MercatorProjector_buildMaps(
		struct RotationWarperBase_MercatorProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" stitching_API
float RotationWarperBase_MercatorProjector_getScale(
		struct RotationWarperBase_MercatorProjectorPtr ptr);

extern "C" stitching_API
void RotationWarperBase_MercatorProjector_setScale(
		struct RotationWarperBase_MercatorProjectorPtr ptr, float val);

extern "C" stitching_API
struct TensorPlusPoint RotationWarperBase_MercatorProjector_warp(
		struct RotationWarperBase_MercatorProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

extern "C" stitching_API
struct TensorWrapper RotationWarperBase_MercatorProjector_warpBackward(
		struct RotationWarperBase_MercatorProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

extern "C" stitching_API
struct Point2fWrapper RotationWarperBase_MercatorProjector_warpPoint(
		struct RotationWarperBase_MercatorProjectorPtr ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

extern "C" stitching_API
struct RectWrapper RotationWarperBase_MercatorProjector_warpRoi(
		struct RotationWarperBase_MercatorProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

//RotationWarperBase_PaniniPortraitProjector

extern "C"
struct RotationWarperBase_PaniniPortraitProjectorPtr {
	void *ptr;

	inline detail::RotationWarperBase<detail::PaniniPortraitProjector> * operator->() {
		return static_cast<detail::RotationWarperBase<detail::PaniniPortraitProjector>  *>(ptr);
	}
	inline RotationWarperBase_PaniniPortraitProjectorPtr(
			detail::RotationWarperBase<detail::PaniniPortraitProjector> *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct RotationWarperBase_PaniniPortraitProjectorPtr RotationWarperBase_PaniniPortraitProjector_ctor();

extern "C" stitching_API
struct TensorArrayPlusRect RotationWarperBase_PaniniPortraitProjector_buildMaps(
		struct RotationWarperBase_PaniniPortraitProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" stitching_API
float RotationWarperBase_PaniniPortraitProjector_getScale(
		struct RotationWarperBase_PaniniPortraitProjectorPtr ptr);

extern "C" stitching_API
void RotationWarperBase_PaniniPortraitProjector_setScale(
		struct RotationWarperBase_PaniniPortraitProjectorPtr ptr, float val);

extern "C" stitching_API
struct TensorPlusPoint RotationWarperBase_PaniniPortraitProjector_warp(
		struct RotationWarperBase_PaniniPortraitProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

extern "C" stitching_API
struct TensorWrapper RotationWarperBase_PaniniPortraitProjector_warpBackward(
		struct RotationWarperBase_PaniniPortraitProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

extern "C" stitching_API
struct Point2fWrapper RotationWarperBase_PaniniPortraitProjector_warpPoint(
		struct RotationWarperBase_PaniniPortraitProjectorPtr ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

extern "C" stitching_API
struct RectWrapper RotationWarperBase_PaniniPortraitProjector_warpRoi(
		struct RotationWarperBase_PaniniPortraitProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

//RotationWarperBase_PaniniProjector

extern "C"
struct RotationWarperBase_PaniniProjectorPtr {
	void *ptr;

	inline detail::RotationWarperBase<detail::PaniniProjector> * operator->() {
		return static_cast<detail::RotationWarperBase<detail::PaniniProjector>  *>(ptr);
	}
	inline RotationWarperBase_PaniniProjectorPtr(
			detail::RotationWarperBase<detail::PaniniProjector> *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct RotationWarperBase_PaniniProjectorPtr RotationWarperBase_PaniniProjector_ctor();

extern "C" stitching_API
struct TensorArrayPlusRect RotationWarperBase_PaniniProjector_buildMaps(
		struct RotationWarperBase_PaniniProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" stitching_API
float RotationWarperBase_PaniniProjector_getScale(
		struct RotationWarperBase_PaniniProjectorPtr ptr);

extern "C" stitching_API
void RotationWarperBase_PaniniProjector_setScale(
		struct RotationWarperBase_PaniniProjectorPtr ptr, float val);

extern "C" stitching_API
struct TensorPlusPoint RotationWarperBase_PaniniProjector_warp(
		struct RotationWarperBase_PaniniProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

extern "C" stitching_API
struct TensorWrapper RotationWarperBase_PaniniProjector_warpBackward(
		struct RotationWarperBase_PaniniProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

extern "C" stitching_API
struct Point2fWrapper RotationWarperBase_PaniniProjector_warpPoint(
		struct RotationWarperBase_PaniniProjectorPtr ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

extern "C" stitching_API
struct RectWrapper RotationWarperBase_PaniniProjector_warpRoi(
		struct RotationWarperBase_PaniniProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

//RotationWarperBase_PlanePortraitProjector

extern "C"
struct RotationWarperBase_PlanePortraitProjectorPtr {
	void *ptr;

	inline detail::RotationWarperBase<detail::PlanePortraitProjector> * operator->() {
		return static_cast<detail::RotationWarperBase<detail::PlanePortraitProjector>  *>(ptr);
	}
	inline RotationWarperBase_PlanePortraitProjectorPtr(
			detail::RotationWarperBase<detail::PlanePortraitProjector> *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct RotationWarperBase_PlanePortraitProjectorPtr RotationWarperBase_PlanePortraitProjector_ctor();

extern "C" stitching_API
struct TensorArrayPlusRect RotationWarperBase_PlanePortraitProjector_buildMaps(
		struct RotationWarperBase_PlanePortraitProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" stitching_API
float RotationWarperBase_PlanePortraitProjector_getScale(
		struct RotationWarperBase_PlanePortraitProjectorPtr ptr);

extern "C" stitching_API
void RotationWarperBase_PlanePortraitProjector_setScale(
		struct RotationWarperBase_PlanePortraitProjectorPtr ptr, float val);

extern "C" stitching_API
struct TensorPlusPoint RotationWarperBase_PlanePortraitProjector_warp(
		struct RotationWarperBase_PlanePortraitProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

extern "C" stitching_API
struct TensorWrapper RotationWarperBase_PlanePortraitProjector_warpBackward(
		struct RotationWarperBase_PlanePortraitProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

extern "C" stitching_API
struct Point2fWrapper RotationWarperBase_PlanePortraitProjector_warpPoint(
		struct RotationWarperBase_PlanePortraitProjectorPtr ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

extern "C" stitching_API
struct RectWrapper RotationWarperBase_PlanePortraitProjector_warpRoi(
		struct RotationWarperBase_PlanePortraitProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

//RotationWarperBase_PlaneProjector

extern "C"
struct RotationWarperBase_PlaneProjectorPtr {
	void *ptr;

	inline detail::RotationWarperBase<detail::PlaneProjector> * operator->() {
		return static_cast<detail::RotationWarperBase<detail::PlaneProjector>  *>(ptr);
	}
	inline RotationWarperBase_PlaneProjectorPtr(
			detail::RotationWarperBase<detail::PlaneProjector> *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct RotationWarperBase_PlaneProjectorPtr RotationWarperBase_PlaneProjector_ctor();

extern "C" stitching_API
struct TensorArrayPlusRect RotationWarperBase_PlaneProjector_buildMaps(
		struct RotationWarperBase_PlaneProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" stitching_API
float RotationWarperBase_PlaneProjector_getScale(
		struct RotationWarperBase_PlaneProjectorPtr ptr);

extern "C" stitching_API
void RotationWarperBase_PlaneProjector_setScale(
		struct RotationWarperBase_PlaneProjectorPtr ptr, float val);

extern "C" stitching_API
struct TensorPlusPoint RotationWarperBase_PlaneProjector_warp(
		struct RotationWarperBase_PlaneProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

extern "C" stitching_API
struct TensorWrapper RotationWarperBase_PlaneProjector_warpBackward(
		struct RotationWarperBase_PlaneProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

extern "C" stitching_API
struct Point2fWrapper RotationWarperBase_PlaneProjector_warpPoint(
		struct RotationWarperBase_PlaneProjectorPtr ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

extern "C" stitching_API
struct RectWrapper RotationWarperBase_PlaneProjector_warpRoi(
		struct RotationWarperBase_PlaneProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

//RotationWarperBase_SphericalPortraitProjector

extern "C"
struct RotationWarperBase_SphericalPortraitProjectorPtr {
	void *ptr;

	inline detail::RotationWarperBase<detail::SphericalPortraitProjector> * operator->() {
		return static_cast<detail::RotationWarperBase<detail::SphericalPortraitProjector>  *>(ptr);
	}
	inline RotationWarperBase_SphericalPortraitProjectorPtr(
			detail::RotationWarperBase<detail::SphericalPortraitProjector> *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct RotationWarperBase_SphericalPortraitProjectorPtr RotationWarperBase_SphericalPortraitProjector_ctor();

extern "C" stitching_API
struct TensorArrayPlusRect RotationWarperBase_SphericalPortraitProjector_buildMaps(
		struct RotationWarperBase_SphericalPortraitProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" stitching_API
float RotationWarperBase_SphericalPortraitProjector_getScale(
		struct RotationWarperBase_SphericalPortraitProjectorPtr ptr);

extern "C" stitching_API
void RotationWarperBase_SphericalPortraitProjector_setScale(
		struct RotationWarperBase_SphericalPortraitProjectorPtr ptr, float val);

extern "C" stitching_API
struct TensorPlusPoint RotationWarperBase_SphericalPortraitProjector_warp(
		struct RotationWarperBase_SphericalPortraitProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

extern "C" stitching_API
struct TensorWrapper RotationWarperBase_SphericalPortraitProjector_warpBackward(
		struct RotationWarperBase_SphericalPortraitProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

extern "C" stitching_API
struct Point2fWrapper RotationWarperBase_SphericalPortraitProjector_warpPoint(
		struct RotationWarperBase_SphericalPortraitProjectorPtr ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

extern "C" stitching_API
struct RectWrapper RotationWarperBase_SphericalPortraitProjector_warpRoi(
		struct RotationWarperBase_SphericalPortraitProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

//RotationWarperBase_SphericalProjector

extern "C"
struct RotationWarperBase_SphericalProjectorPtr {
	void *ptr;

	inline detail::RotationWarperBase<detail::SphericalProjector> * operator->() {
		return static_cast<detail::RotationWarperBase<detail::SphericalProjector>  *>(ptr);
	}
	inline RotationWarperBase_SphericalProjectorPtr(
			detail::RotationWarperBase<detail::SphericalProjector> *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct RotationWarperBase_SphericalProjectorPtr RotationWarperBase_SphericalProjector_ctor();

extern "C" stitching_API
struct TensorArrayPlusRect RotationWarperBase_SphericalProjector_buildMaps(
		struct RotationWarperBase_SphericalProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" stitching_API
float RotationWarperBase_SphericalProjector_getScale(
		struct RotationWarperBase_SphericalProjectorPtr ptr);

extern "C" stitching_API
void RotationWarperBase_SphericalProjector_setScale(
		struct RotationWarperBase_SphericalProjectorPtr ptr, float val);

extern "C" stitching_API
struct TensorPlusPoint RotationWarperBase_SphericalProjector_warp(
		struct RotationWarperBase_SphericalProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

extern "C" stitching_API
struct TensorWrapper RotationWarperBase_SphericalProjector_warpBackward(
		struct RotationWarperBase_SphericalProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

extern "C" stitching_API
struct Point2fWrapper RotationWarperBase_SphericalProjector_warpPoint(
		struct RotationWarperBase_SphericalProjectorPtr ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

extern "C" stitching_API
struct RectWrapper RotationWarperBase_SphericalProjector_warpRoi(
		struct RotationWarperBase_SphericalProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

//RotationWarperBase_StereographicProjector

extern "C"
struct RotationWarperBase_StereographicProjectorPtr {
	void *ptr;

	inline detail::RotationWarperBase<detail::StereographicProjector> * operator->() {
		return static_cast<detail::RotationWarperBase<detail::StereographicProjector>  *>(ptr);
	}
	inline RotationWarperBase_StereographicProjectorPtr(
			detail::RotationWarperBase<detail::StereographicProjector> *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct RotationWarperBase_StereographicProjectorPtr RotationWarperBase_StereographicProjector_ctor();

extern "C" stitching_API
struct TensorArrayPlusRect RotationWarperBase_StereographicProjector_buildMaps(
		struct RotationWarperBase_StereographicProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" stitching_API
float RotationWarperBase_StereographicProjector_getScale(
		struct RotationWarperBase_StereographicProjectorPtr ptr);

extern "C" stitching_API
void RotationWarperBase_StereographicProjector_setScale(
		struct RotationWarperBase_StereographicProjectorPtr ptr, float val);

extern "C" stitching_API
struct TensorPlusPoint RotationWarperBase_StereographicProjector_warp(
		struct RotationWarperBase_StereographicProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

extern "C" stitching_API
struct TensorWrapper RotationWarperBase_StereographicProjector_warpBackward(
		struct RotationWarperBase_StereographicProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

extern "C" stitching_API
struct Point2fWrapper RotationWarperBase_StereographicProjector_warpPoint(
		struct RotationWarperBase_StereographicProjectorPtr ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

extern "C" stitching_API
struct RectWrapper RotationWarperBase_StereographicProjector_warpRoi(
		struct RotationWarperBase_StereographicProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

//RotationWarperBase_TransverseMercatorProjector

extern "C"
struct RotationWarperBase_TransverseMercatorProjectorPtr {
	void *ptr;

	inline detail::RotationWarperBase<detail::TransverseMercatorProjector> * operator->() {
		return static_cast<detail::RotationWarperBase<detail::TransverseMercatorProjector>  *>(ptr);
	}
	inline RotationWarperBase_TransverseMercatorProjectorPtr(
			detail::RotationWarperBase<detail::TransverseMercatorProjector> *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct RotationWarperBase_TransverseMercatorProjectorPtr RotationWarperBase_TransverseMercatorProjector_ctor();

extern "C" stitching_API
struct TensorArrayPlusRect RotationWarperBase_TransverseMercatorProjector_buildMaps(
		struct RotationWarperBase_TransverseMercatorProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" stitching_API
float RotationWarperBase_TransverseMercatorProjector_getScale(
		struct RotationWarperBase_TransverseMercatorProjectorPtr ptr);

extern "C" stitching_API
void RotationWarperBase_TransverseMercatorProjector_setScale(
		struct RotationWarperBase_TransverseMercatorProjectorPtr ptr, float val);

extern "C" stitching_API
struct TensorPlusPoint RotationWarperBase_TransverseMercatorProjector_warp(
		struct RotationWarperBase_TransverseMercatorProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

extern "C" stitching_API
struct TensorWrapper RotationWarperBase_TransverseMercatorProjector_warpBackward(
		struct RotationWarperBase_TransverseMercatorProjectorPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct SizeWrapper dst_size,
		struct TensorWrapper dst);

extern "C" stitching_API
struct Point2fWrapper RotationWarperBase_TransverseMercatorProjector_warpPoint(
		struct RotationWarperBase_TransverseMercatorProjectorPtr ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

extern "C" stitching_API
struct RectWrapper RotationWarperBase_TransverseMercatorProjector_warpRoi(
		struct RotationWarperBase_TransverseMercatorProjectorPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

//WarperCreator

extern "C"
struct WarperCreatorPtr {
	void *ptr;

	inline cv::WarperCreator * operator->() { return static_cast<cv::WarperCreator *>(ptr);
	}
	inline WarperCreatorPtr(cv::WarperCreator *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
void WarperCreator_dtor(
		struct WarperCreatorPtr ptr);

extern "C" stitching_API
struct RotationWarperPtr WarperCreator_create(
		struct WarperCreatorPtr ptr, float scale);

//CompressedRectilinearPortraitWarper

extern "C"
struct CompressedRectilinearPortraitWarperPtr {
	void *ptr;

	inline cv::CompressedRectilinearPortraitWarper * operator->() {
		return static_cast<cv::CompressedRectilinearPortraitWarper *>(ptr);
	}
	inline CompressedRectilinearPortraitWarperPtr(cv::CompressedRectilinearPortraitWarper *ptr) {
		this->ptr = ptr;
	}
};

extern "C" stitching_API
struct CompressedRectilinearPortraitWarperPtr CompressedRectilinearPortraitWarper_ctor(
		float A, float B);

extern "C" stitching_API
struct RotationWarperPtr CompressedRectilinearPortraitWarper_create(
		struct CompressedRectilinearPortraitWarperPtr ptr, float scale);

//CompressedRectilinearWarper

extern "C"
struct CompressedRectilinearWarperPtr {
	void *ptr;

	inline cv::CompressedRectilinearWarper * operator->() {
		return static_cast<cv::CompressedRectilinearWarper *>(ptr);
	}
	inline CompressedRectilinearWarperPtr(cv::CompressedRectilinearWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct CompressedRectilinearWarperPtr CompressedRectilinearWarper_ctor(
		float A, float B);

extern "C" stitching_API
struct RotationWarperPtr CompressedRectilinearWarper_create(
		struct CompressedRectilinearWarperPtr ptr, float scale);

//CylindricalWarper

extern "C"
struct CylindricalWarperPtr {
	void *ptr;

	inline cv::CylindricalWarper * operator->() {
		return static_cast<cv::CylindricalWarper *>(ptr);
	}
	inline CylindricalWarperPtr(cv::CylindricalWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct CylindricalWarperPtr CylindricalWarper_ctor();

extern "C" stitching_API
struct RotationWarperPtr CylindricalWarper_create(
		struct CylindricalWarperPtr ptr, float scale);

//FisheyeWarper

extern "C"
struct FisheyeWarperPtr {
	void *ptr;

	inline cv::FisheyeWarper * operator->() {
		return static_cast<cv::FisheyeWarper *>(ptr);
	}
	inline FisheyeWarperPtr(cv::FisheyeWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct FisheyeWarperPtr FisheyeWarper_ctor();

extern "C" stitching_API
struct RotationWarperPtr FisheyeWarper_create(
		struct FisheyeWarperPtr ptr, float scale);

//MercatorWarper

extern "C" stitching_API
struct MercatorWarperPtr {
	void *ptr;

	inline cv::MercatorWarper * operator->() {
		return static_cast<cv::MercatorWarper *>(ptr);
	}
	inline MercatorWarperPtr(cv::MercatorWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct MercatorWarperPtr MercatorWarper_ctor();

extern "C" stitching_API
struct RotationWarperPtr MercatorWarper_create(
		struct MercatorWarperPtr ptr, float scale);

//PaniniPortraitWarper

extern "C"
struct PaniniPortraitWarperPtr {
	void *ptr;

	inline cv::PaniniPortraitWarper * operator->() {
		return static_cast<cv::PaniniPortraitWarper *>(ptr);
	}
	inline PaniniPortraitWarperPtr(cv::PaniniPortraitWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct PaniniPortraitWarperPtr PaniniPortraitWarper_ctor(
		float A, float B);

extern "C" stitching_API
struct RotationWarperPtr PaniniPortraitWarper_create(
		struct PaniniPortraitWarperPtr ptr, float scale);

//PaniniWarper

extern "C"
struct PaniniWarperPtr {
	void *ptr;

	inline cv::PaniniWarper * operator->() {
		return static_cast<cv::PaniniWarper *>(ptr);
	}
	inline PaniniWarperPtr(cv::PaniniWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct PaniniWarperPtr PaniniWarper_ctor(
		float A, float B);

extern "C" stitching_API
struct RotationWarperPtr PaniniWarper_create(
		struct PaniniWarperPtr ptr, float scale);

//PlaneWarper

extern "C"
struct PlaneWarperPtr {
	void *ptr;

	inline cv::PlaneWarper * operator->() {
		return static_cast<cv::PlaneWarper *>(ptr);
	}
	inline PlaneWarperPtr(cv::PlaneWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct PlaneWarperPtr PlaneWarper_ctor();

extern "C" stitching_API
struct RotationWarperPtr PlaneWarper_create(
		struct PlaneWarperPtr ptr, float scale);

//SphericalWarper

extern "C"
struct SphericalWarperPtr {
	void *ptr;

	inline cv::SphericalWarper * operator->() {
		return static_cast<cv::SphericalWarper *>(ptr);
	}
	inline SphericalWarperPtr(cv::SphericalWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct SphericalWarperPtr SphericalWarper_ctor();

extern "C" stitching_API
struct RotationWarperPtr SphericalWarper_create(
		struct SphericalWarperPtr ptr, float scale);

//StereographicWarper

extern "C"
struct StereographicWarperPtr {
	void *ptr;

	inline cv::StereographicWarper * operator->() {
		return static_cast<cv::StereographicWarper *>(ptr);
	}
	inline StereographicWarperPtr(cv::StereographicWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct StereographicWarperPtr StereographicWarper_ctor();

extern "C" stitching_API
struct RotationWarperPtr StereographicWarper_create(
		struct StereographicWarperPtr ptr, float scale);

//TransverseMercatorWarper

extern "C"
struct TransverseMercatorWarperPtr {
	void *ptr;

	inline cv::TransverseMercatorWarper * operator->() {
		return static_cast<cv::TransverseMercatorWarper *>(ptr);
	}
	inline TransverseMercatorWarperPtr(cv::TransverseMercatorWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct TransverseMercatorWarperPtr TransverseMercatorWarper_ctor();

extern "C" stitching_API
struct RotationWarperPtr TransverseMercatorWarper_create(
		struct TransverseMercatorWarperPtr ptr, float scale);

//detail_CompressedRectilinearPortraitWarper

extern "C"
struct detail_CompressedRectilinearPortraitWarperPtr {
	void *ptr;

	inline cv::detail::CompressedRectilinearPortraitWarper * operator->() {
		return static_cast<cv::detail::CompressedRectilinearPortraitWarper *>(ptr);
	}
	inline detail_CompressedRectilinearPortraitWarperPtr(cv::detail::CompressedRectilinearPortraitWarper *ptr) {
		this->ptr = ptr;
	}
};

extern "C" stitching_API
struct detail_CompressedRectilinearPortraitWarperPtr detail_CompressedRectilinearPortraitWarper_ctor(
		float scale, float A, float B);

//detail_CompressedRectilinearWarper

extern "C"
struct detail_CompressedRectilinearWarperPtr {
	void *ptr;

	inline cv::detail::CompressedRectilinearWarper * operator->() {
		return static_cast<cv::detail::CompressedRectilinearWarper *>(ptr);
	}
	inline detail_CompressedRectilinearWarperPtr(cv::detail::CompressedRectilinearWarper *ptr) {
		this->ptr = ptr;
	}
};

extern "C" stitching_API
struct detail_CompressedRectilinearWarperPtr detail_CompressedRectilinearWarper_ctor(
		float scale, float A, float B);

//detail_CylindricalPortraitWarper

extern "C"
struct detail_CylindricalPortraitWarperPtr {
	void *ptr;

	inline cv::detail::CylindricalPortraitWarper * operator->() {
		return static_cast<cv::detail::CylindricalPortraitWarper *>(ptr);
	}
	inline detail_CylindricalPortraitWarperPtr(cv::detail::CylindricalPortraitWarper *ptr) {
		this->ptr = ptr;
	}
};

extern "C" stitching_API
struct detail_CylindricalPortraitWarperPtr detail_CylindricalPortraitWarper_ctor(
		float scale);

//detail_CylindricalWarper

extern "C"
struct detail_CylindricalWarperPtr {
	void *ptr;

	inline cv::detail::CylindricalWarper * operator->() {
		return static_cast<cv::detail::CylindricalWarper *>(ptr);
	}
	inline detail_CylindricalWarperPtr(cv::detail::CylindricalWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct detail_CylindricalWarperPtr detail_CylindricalWarper_ctor(
		float scale);

extern "C" stitching_API
struct TensorArrayPlusRect detail_CylindricalWarper_buildMaps(
		struct detail_CylindricalWarperPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" stitching_API
struct TensorPlusPoint detail_CylindricalWarper_warp(
		struct detail_CylindricalWarperPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct TensorWrapper dst);

//detail_CylindricalWarperGpu

//TODO need to add two cuda functions

extern "C"
struct detail_CylindricalWarperGpuPtr {
	void *ptr;

	inline cv::detail::CylindricalWarperGpu * operator->() {
		return static_cast<cv::detail::CylindricalWarperGpu *>(ptr);
	}
	inline detail_CylindricalWarperGpuPtr(cv::detail::CylindricalWarperGpu *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct detail_CylindricalWarperGpuPtr detail_CylindricalWarperGpu_ctor(
		float scale);

extern "C" stitching_API
struct TensorArrayPlusRect detail_CylindricalWarperGpu_buildMaps(
		struct detail_CylindricalWarperGpuPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" stitching_API
struct TensorPlusPoint detail_CylindricalWarperGpu_warp(
		struct detail_CylindricalWarperGpuPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct TensorWrapper dst);

//detail_FisheyeWarper

extern "C"
struct detail_FisheyeWarperPtr {
	void *ptr;

	inline cv::detail::FisheyeWarper * operator->() {
		return static_cast<cv::detail::FisheyeWarper *>(ptr);
	}
	inline detail_FisheyeWarperPtr(cv::detail::FisheyeWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct detail_FisheyeWarperPtr detail_FisheyeWarper_ctor(
		float scale);

//detail_MercatorWarper

extern "C"
struct detail_MercatorWarperPtr {
	void *ptr;

	inline cv::detail::MercatorWarper * operator->() {
		return static_cast<cv::detail::MercatorWarper *>(ptr);
	}
	inline detail_MercatorWarperPtr(cv::detail::MercatorWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct detail_MercatorWarperPtr detail_MercatorWarper_ctor(
		float scale);

//detail_PaniniPortraitWarper

extern "C"
struct detail_PaniniPortraitWarperPtr {
	void *ptr;

	inline cv::detail::PaniniPortraitWarper * operator->() {
		return static_cast<cv::detail::PaniniPortraitWarper *>(ptr);
	}
	inline detail_PaniniPortraitWarperPtr(cv::detail::PaniniPortraitWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct detail_PaniniPortraitWarperPtr detail_PaniniPortraitWarper_ctor(
		float scale, float A, float B);

//detail_PaniniWarper

extern "C"
struct detail_PaniniWarperPtr {
	void *ptr;

	inline cv::detail::PaniniWarper * operator->() {
		return static_cast<cv::detail::PaniniWarper *>(ptr);
	}
	inline detail_PaniniWarperPtr(cv::detail::PaniniWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct detail_PaniniWarperPtr detail_PaniniWarper_ctor(
		float scale, float A, float B);

//detail_PlanePortraitWarper

extern "C"
struct detail_PlanePortraitWarperPtr {
	void *ptr;

	inline cv::detail::PlanePortraitWarper * operator->() {
		return static_cast<cv::detail::PlanePortraitWarper *>(ptr);
	}
	inline detail_PlanePortraitWarperPtr(cv::detail::PlanePortraitWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct detail_PlanePortraitWarperPtr detail_PlanePortraitWarper_ctor(
		float scale);

//detail_PlaneWarper

extern "C"
struct detail_PlaneWarperPtr {
	void *ptr;

	inline cv::detail::PlaneWarper * operator->() {
		return static_cast<cv::detail::PlaneWarper *>(ptr);
	}
	inline detail_PlaneWarperPtr(cv::detail::PlaneWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct detail_PlaneWarperPtr detail_PlaneWarper_ctor(
		float scale);

extern "C" stitching_API
struct TensorArrayPlusRect detail_PlaneWarper_buildMaps2(
		struct detail_PlaneWarperPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" stitching_API
struct TensorArrayPlusRect detail_PlaneWarper_buildMaps(
		struct detail_PlaneWarperPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper T, struct TensorWrapper xmap,
		struct TensorWrapper ymap);

extern "C" stitching_API
struct TensorPlusPoint detail_PlaneWarper_warp(
		struct detail_PlaneWarperPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper T, int interp_mode,
		int border_mode, struct TensorWrapper dst);

extern "C" stitching_API
struct TensorPlusPoint detail_PlaneWarper_warp2(
		struct detail_PlaneWarperPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,
		int interp_mode, int border_mode, struct TensorWrapper dst);

extern "C" stitching_API
struct Point2fWrapper detail_PlaneWarper_warpPoint(
		struct detail_PlaneWarperPtr ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R, struct TensorWrapper T);

extern "C" stitching_API
struct Point2fWrapper detail_PlaneWarper_warpPoint2(
		struct detail_PlaneWarperPtr ptr, struct Point2fWrapper pt,
		struct TensorWrapper K, struct TensorWrapper R);

extern "C" stitching_API
struct RectWrapper detail_PlaneWarper_warpRoi(
		struct detail_PlaneWarperPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper T);

extern "C" stitching_API
struct RectWrapper detail_PlaneWarper_warpRoi2(
		struct detail_PlaneWarperPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R);

//detail_SphericalPortraitWarper

extern "C"
struct detail_SphericalPortraitWarperPtr {
	void *ptr;

	inline cv::detail::SphericalPortraitWarper * operator->() {
		return static_cast<cv::detail::SphericalPortraitWarper *>(ptr);
	}
	inline detail_SphericalPortraitWarperPtr(cv::detail::SphericalPortraitWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct detail_SphericalPortraitWarperPtr detail_SphericalPortraitWarper_ctor(
		float scale);

//detail_SphericalWarper

extern "C"
struct detail_SphericalWarperPtr {
	void *ptr;

	inline cv::detail::SphericalWarper * operator->() {
		return static_cast<cv::detail::SphericalWarper *>(ptr);
	}
	inline detail_SphericalWarperPtr(cv::detail::SphericalWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct detail_SphericalWarperPtr detail_SphericalWarper_ctor(
		float scale);

extern "C" stitching_API
struct TensorArrayPlusRect detail_SphericalWarper_buildMaps(
		struct detail_SphericalWarperPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" stitching_API
struct TensorPlusPoint detail_SphericalWarper_warp(
		struct detail_SphericalWarperPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

//detail_SphericalWarperGpu

//TODO need to add two cuda function

extern "C"
struct detail_SphericalWarperGpuPtr {
	void *ptr;

	inline cv::detail::SphericalWarperGpu * operator->() {
		return static_cast<cv::detail::SphericalWarperGpu *>(ptr);
	}
	inline detail_SphericalWarperGpuPtr(cv::detail::SphericalWarperGpu *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct detail_SphericalWarperGpuPtr detail_SphericalWarperGpu_ctor(
		float scale);

extern "C" stitching_API
struct TensorArrayPlusRect detail_SphericalWarperGpu_buildMaps(
		struct detail_SphericalWarperGpuPtr ptr, struct SizeWrapper src_size,
		struct TensorWrapper K, struct TensorWrapper R,
		struct TensorWrapper xmap, struct TensorWrapper ymap);

extern "C" stitching_API
struct TensorPlusPoint detail_SphericalWarperGpu_warp(
		struct detail_SphericalWarperGpuPtr ptr, struct TensorWrapper src,
		struct TensorWrapper K, struct TensorWrapper R,	int interp_mode,
		int border_mode, struct TensorWrapper dst);

//detail_StereographicWarper

extern "C"
struct detail_StereographicWarperPtr {
	void *ptr;

	inline cv::detail::StereographicWarper * operator->() {
		return static_cast<cv::detail::StereographicWarper *>(ptr);
	}
	inline detail_StereographicWarperPtr(cv::detail::StereographicWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct detail_StereographicWarperPtr detail_StereographicWarper_ctor(
		float scale);

//detail_TransverseMercatorWarper

extern "C" stitching_API
struct detail_TransverseMercatorWarperPtr {
	void *ptr;

	inline cv::detail::TransverseMercatorWarper * operator->() {
		return static_cast<cv::detail::TransverseMercatorWarper *>(ptr);
	}
	inline detail_TransverseMercatorWarperPtr(cv::detail::TransverseMercatorWarper *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct detail_TransverseMercatorWarperPtr detail_TransverseMercatorWarper_ctor(
		float scale);


//************************Seam Estimation******************************


//SeamFinder

extern "C"
struct SeamFinderPtr {
	void *ptr;

	inline cv::detail::SeamFinder * operator->() {
		return static_cast<cv::detail::SeamFinder *>(ptr);
	}
	inline SeamFinderPtr(cv::detail::SeamFinder *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
void SeamFinder_dtor(
		struct SeamFinderPtr ptr);

extern "C" stitching_API
void SeamFinder_find(
		struct SeamFinderPtr ptr, struct TensorArray src, struct PointArray corners, struct TensorArray masks);

//DpSeamFinder

extern "C" 
struct DpSeamFinderPtr {
	void *ptr;

	inline cv::detail::DpSeamFinder * operator->() {
		return static_cast<cv::detail::DpSeamFinder *>(ptr);
	}
	inline DpSeamFinderPtr(cv::detail::DpSeamFinder *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct DpSeamFinderPtr DpSeamFinder_ctor(int costFunc);

extern "C" stitching_API
int DpSeamFinder_costFunction(
		struct DpSeamFinderPtr ptr);

extern "C" stitching_API
void DpSeamFinder_find(
		struct DpSeamFinderPtr ptr, struct TensorArray src, struct PointArray corners, struct TensorArray masks);

extern "C" stitching_API
void DpSeamFinder_setCostFunction(
		struct DpSeamFinderPtr ptr, int val);

//GraphCutSeamFinder

extern "C"
struct GraphCutSeamFinderPtr {
	void *ptr;

	inline cv::detail::GraphCutSeamFinder * operator->() {
		return static_cast<cv::detail::GraphCutSeamFinder *>(ptr);
	}
	inline GraphCutSeamFinderPtr(cv::detail::GraphCutSeamFinder *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct GraphCutSeamFinderPtr GraphCutSeamFinder_ctor(
		int cost_type, float terminal_cost, float bad_region_penalty);

extern "C" stitching_API
void GraphCutSeamFinder_dtor(
		struct GraphCutSeamFinderPtr ptr);

extern "C" stitching_API
void GraphCutSeamFinder_find(
		struct GraphCutSeamFinderPtr ptr, struct TensorArray src,
		struct PointArray corners, struct TensorArray masks);

//NoSeamFinder

extern "C"
struct NoSeamFinderPtr {
	void *ptr;

	inline cv::detail::NoSeamFinder * operator->() {
		return static_cast<cv::detail::NoSeamFinder *>(ptr);
	}
	inline NoSeamFinderPtr(cv::detail::NoSeamFinder *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct NoSeamFinderPtr NoSeamFinder_ctor();

extern "C" stitching_API
void NoSeamFinder_find(
		struct NoSeamFinderPtr ptr, struct TensorArray src,
		struct PointArray corners, struct TensorArray masks);

//PairwiseSeamFinder

extern "C"
struct PairwiseSeamFinderPtr {
	void *ptr;

	inline cv::detail::PairwiseSeamFinder * operator->() {
		return static_cast<cv::detail::PairwiseSeamFinder *>(ptr);
	}
	inline PairwiseSeamFinderPtr(cv::detail::PairwiseSeamFinder *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
void PairwiseSeamFinder_find(
		struct PairwiseSeamFinderPtr ptr, struct TensorArray src,
		struct PointArray corners, struct TensorArray masks);

//VoronoiSeamFinder

extern "C"
struct VoronoiSeamFinderPtr {
	void *ptr;

	inline cv::detail::VoronoiSeamFinder * operator->() {
		return static_cast<cv::detail::VoronoiSeamFinder *>(ptr);
	}
	inline VoronoiSeamFinderPtr(cv::detail::VoronoiSeamFinder *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct VoronoiSeamFinderPtr VoronoiSeamFinder();

extern "C" stitching_API
void VoronoiSeamFinder_find(
		struct VoronoiSeamFinderPtr ptr, struct TensorArray src,
		struct PointArray corners, struct TensorArray masks);

extern "C" stitching_API
void VoronoiSeamFinder_find2(
		struct VoronoiSeamFinderPtr ptr, struct SizeArray size,
		struct PointArray corners, struct TensorArray masks);


//************************ExposureCompensator*******************************


//ExposureCompensator

extern "C"
struct ExposureCompensatorPtr {
	void *ptr;

	inline cv::detail::ExposureCompensator * operator->() {
		return static_cast<cv::detail::ExposureCompensator *>(ptr);
	}
	inline ExposureCompensatorPtr(cv::detail::ExposureCompensator *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct ExposureCompensatorPtr ExposureCompensator_ctor(
		int type);

extern "C" stitching_API
void ExposureCompensator_dtor(
		struct ExposureCompensatorPtr ptr);

extern "C" stitching_API
void  ExposureCompensator_apply(
		struct ExposureCompensatorPtr ptr, int index, struct PointWrapper corner,
		struct TensorWrapper image, struct TensorWrapper mask);

extern "C" stitching_API
void ExposureCompensator_feed(
		struct ExposureCompensatorPtr ptr, struct PointArray corners,
		struct TensorArray images, struct TensorArray masks);

//BlocksGainCompensator

extern "C"
struct BlocksGainCompensatorPtr {
	void *ptr;

	inline cv::detail::BlocksGainCompensator * operator->() {
		return static_cast<cv::detail::BlocksGainCompensator *>(ptr);
	}
	inline BlocksGainCompensatorPtr(cv::detail::BlocksGainCompensator *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct BlocksGainCompensatorPtr BlocksGainCompensator_ctor(
		int bl_width, int bl_height);

extern "C" stitching_API
void  BlocksGainCompensator_apply(
		struct BlocksGainCompensatorPtr ptr, int index, struct PointWrapper corner,
		struct TensorWrapper image, struct TensorWrapper mask);

extern "C" stitching_API
void BlocksGainCompensator_feed(
		struct BlocksGainCompensatorPtr ptr, struct PointArray corners,
		struct TensorArray images, struct TensorArray mat, struct UCharArray chr);

//GainCompensator

extern "C"
struct GainCompensatorPtr {
	void *ptr;

	inline cv::detail::GainCompensator * operator->() {
		return static_cast<cv::detail::GainCompensator *>(ptr);
	}
	inline GainCompensatorPtr(cv::detail::GainCompensator *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct GainCompensatorPtr GainCompensator_ctor();

extern "C" stitching_API
void  GainCompensator_apply(
		struct GainCompensatorPtr ptr, int index, struct PointWrapper corner,
		struct TensorWrapper image, struct TensorWrapper mask);

extern "C" stitching_API
void GainCompensator_feed(
		struct GainCompensatorPtr ptr, struct PointArray corners,
		struct TensorArray images, struct TensorArray mat, struct UCharArray chr);

extern "C" stitching_API
struct TensorWrapper GainCompensator_gains(
		struct GainCompensatorPtr ptr);

//NoExposureCompensator

extern "C"
struct NoExposureCompensatorPtr {
	void *ptr;

	inline cv::detail::NoExposureCompensator * operator->() {
		return static_cast<cv::detail::NoExposureCompensator *>(ptr);
	}
	inline NoExposureCompensatorPtr(cv::detail::NoExposureCompensator *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct NoExposureCompensatorPtr NoExposureCompensator_ctor();

extern "C" stitching_API
void  NoExposureCompensator_apply(
		struct NoExposureCompensatorPtr ptr, int index, struct PointWrapper corner,
		struct TensorWrapper image, struct TensorWrapper mask);

extern "C" stitching_API
void NoExposureCompensator_feed(
		struct NoExposureCompensatorPtr ptr, struct PointArray corners,
		struct TensorArray images, struct TensorArray mat, struct UCharArray chr);


//*******************************Image Blenders**********************


extern "C" stitching_API
struct TensorArray detail_createLaplacePyr(
		struct TensorWrapper img, int num_levels);

extern "C" stitching_API
struct TensorArray detail_createLaplacePyrGpu(
		struct TensorWrapper img, int num_levels);

extern "C" stitching_API
void detail_createWeightMap(
		struct TensorWrapper mask, float sharpness,
		struct TensorWrapper weight);

extern "C" stitching_API
void detail_normalizeUsingWeightMap(
		struct TensorWrapper weight, struct TensorWrapper src);

extern "C" stitching_API
void detail_restoreImageFromLaplacePyr(
		struct TensorArray pyr);

extern "C" stitching_API
void detail_restoreImageFromLaplacePyrGpu(
		struct TensorArray pyr);

//Blender

extern "C"
struct BlenderPtr {
	void *ptr;

	inline cv::detail::Blender * operator->() {
		return static_cast<cv::detail::Blender *>(ptr);
	}
	inline BlenderPtr(cv::detail::Blender *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct BlenderPtr Blender_ctor(
		int type, bool try_gpu);

extern "C" stitching_API
void Blender_dtor(
		struct BlenderPtr ptr);

extern "C" stitching_API
void Blender_blend(
		struct BlenderPtr ptr, struct TensorWrapper dst,
		struct TensorWrapper dst_mask);

extern "C" stitching_API
void Blender_feed(
		struct BlenderPtr ptr, struct TensorWrapper img,
		struct TensorWrapper mask, struct PointWrapper tl);

extern "C" stitching_API
void Blender_prepare(
		struct BlenderPtr ptr, struct RectWrapper dst_roi);

extern "C" stitching_API
void Blender_prepare2(
		struct BlenderPtr ptr, struct PointArray corners,
		struct SizeArray sizes);

//FeatherBlender

extern "C"
struct FeatherBlenderPtr {
	void *ptr;

	inline cv::detail::FeatherBlender * operator->() {
		return static_cast<cv::detail::FeatherBlender *>(ptr);
	}
	inline FeatherBlenderPtr(cv::detail::FeatherBlender *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct FeatherBlenderPtr FeatherBlender_ctor(
		float sharpness);

extern "C" stitching_API
void FeatherBlender_blend(
		struct FeatherBlenderPtr ptr, struct TensorWrapper dst,
		struct TensorWrapper dst_mask);

extern "C" stitching_API
struct RectWrapper FeatherBlender_createWeightMaps(
		struct FeatherBlenderPtr ptr, struct TensorArray masks,
		struct PointArray corners, struct TensorArray weight_maps);

extern "C" stitching_API
void FeatherBlender_feed(
		struct FeatherBlenderPtr ptr, struct TensorWrapper img,
		struct TensorWrapper mask, struct PointWrapper tl);

extern "C" stitching_API
void FeatherBlender_prepare(
		struct FeatherBlenderPtr ptr, struct RectWrapper dst_roi);

extern "C" stitching_API
void FeatherBlender_setSharpness(
		struct FeatherBlenderPtr ptr, float val);

extern "C" stitching_API
float FeatherBlender_sharpness(
		struct FeatherBlenderPtr ptr);

//MultiBandBlender

extern "C"
struct MultiBandBlenderPtr {
	void *ptr;

	inline cv::detail::MultiBandBlender * operator->() {
		return static_cast<cv::detail::MultiBandBlender *>(ptr);
	}
	inline MultiBandBlenderPtr(cv::detail::MultiBandBlender *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct MultiBandBlenderPtr MultiBandBlender_ctor(
		int try_gpu, int num_bands, int weight_type);

extern "C" stitching_API
void MultiBandBlender_blend(
		struct MultiBandBlenderPtr ptr, struct TensorWrapper dst,
		struct TensorWrapper dst_mask);

extern "C" stitching_API
void MultiBandBlender_feed(
		struct MultiBandBlenderPtr ptr, struct TensorWrapper img,
		struct TensorWrapper mask, struct PointWrapper tl);

extern "C" stitching_API
int MultiBandBlender_numBands(
		struct MultiBandBlenderPtr ptr);

extern "C" stitching_API
void MultiBandBlender_setNumBands(
		struct MultiBandBlenderPtr ptr, int val);

//Stitcher

extern "C"
struct StitcherPtr {
	void *ptr;

	inline cv::Stitcher * operator->() { return static_cast<cv::Stitcher *>(ptr); }
	inline StitcherPtr(cv::Stitcher *ptr) { this->ptr = ptr; }
};

extern "C" stitching_API
struct StitcherPtr Stitcher_ctor(bool try_use_gpu);

extern "C" stitching_API
void Stitcher_dtor(
		struct StitcherPtr ptr);

extern "C" stitching_API
struct BlenderPtr Stitcher_blender(
		struct StitcherPtr ptr);

extern "C" stitching_API
struct BundleAdjusterBasePtr Stitcher_bundleAdjuster(
		struct StitcherPtr ptr);

extern "C" stitching_API
struct ClassArray Stitcher_cameras(
		struct StitcherPtr ptr);

extern "C" stitching_API
struct TensorWrapper Stitcher_component(
		struct StitcherPtr ptr);

extern "C" stitching_API
struct TensorPlusInt Stitcher_composePanorama(
		struct StitcherPtr ptr);

extern "C" stitching_API
struct TensorPlusInt Stitcher_composePanorama2(
		struct StitcherPtr ptr, struct TensorArray images);

extern "C" stitching_API
double Stitcher_compositingResol(
		struct StitcherPtr ptr);

extern "C" stitching_API
struct StitcherPtr Stitcher_createDefault(
		struct StitcherPtr ptr, bool try_use_gpu);

extern "C" stitching_API
int Stitcher_estimateTransform(
		struct StitcherPtr ptr, struct TensorArray images);

//TODO need to add 2nd Stitcher_estimateTransform

extern "C" stitching_API
struct ExposureCompensatorPtr Stitcher_exposureCompensator(
		struct StitcherPtr ptr);

extern "C" stitching_API
struct FeaturesFinderPtr Stitcher_featuresFinder(
		struct StitcherPtr ptr);

extern "C" stitching_API
struct FeaturesMatcherPtr Stitcher_featuresMatcher(
		struct StitcherPtr ptr);

extern "C" stitching_API
struct TensorWrapper Stitcher_matchingMask(
		struct StitcherPtr ptr);

extern "C" stitching_API
double Stitcher_panoConfidenceThresh(
		struct StitcherPtr ptr);

extern "C" stitching_API
double Stitcher_registrationResol(
		struct StitcherPtr ptr);

extern "C" stitching_API
double Stitcher_seamEstimationResol(
		struct StitcherPtr ptr);

extern "C" stitching_API
struct SeamFinderPtr Stitcher_seamFinder(
		struct StitcherPtr ptr);

extern "C" stitching_API
void Stitcher_setBlender(
		struct StitcherPtr ptr, struct BlenderPtr b);

extern "C" stitching_API
void Stitcher_setBundleAdjuster(
		struct StitcherPtr ptr, struct BundleAdjusterBasePtr bundle_adjuster);

extern "C" stitching_API
void Stitcher_setCompositingResol(
		struct StitcherPtr ptr, double resol_mpx);

extern "C" stitching_API
void Stitcher_setExposureCompensator(
		struct StitcherPtr ptr, struct ExposureCompensatorPtr exposure_comp);

extern "C" stitching_API
void Stitcher_setFeaturesFinder(
		struct StitcherPtr ptr, struct FeaturesFinderPtr features_finder);

extern "C" stitching_API
void Stitcher_setFeaturesMatcher(
		struct StitcherPtr ptr, FeaturesMatcherPtr features_matcher);

extern "C" stitching_API
void Stitcher_setMatchingMask(
		struct StitcherPtr ptr, struct TensorWrapper mask);

extern "C" stitching_API
void Stitcher_setPanoConfidenceThresh(
		struct StitcherPtr ptr, double conf_thresh);

extern "C" stitching_API
void Stitcher_setRegistrationResol(
		struct StitcherPtr ptr, double resol_mpx);

extern "C" stitching_API
void Stitcher_setSeamEstimationResol(
		struct StitcherPtr ptr, double resol_mpx);

extern "C" stitching_API
void Stitcher_setSeamFinder(
		struct StitcherPtr ptr, struct SeamFinderPtr seam_finder);

extern "C" stitching_API
void Stitcher_setWarper(
		struct StitcherPtr ptr, struct WarperCreatorPtr creator);

extern "C" stitching_API
void Stitcher_setWaveCorrection(
		struct StitcherPtr ptr, bool flag);

extern "C" stitching_API
void Stitcher_setWaveCorrectKind(
		struct StitcherPtr ptr, int kind);

extern "C" stitching_API
struct TensorPlusInt Stitcher_stitch(
		struct StitcherPtr ptr, struct TensorArray images);

extern "C" stitching_API
struct WarperCreatorPtr Stitcher_warper(
		struct StitcherPtr ptr);

extern "C" stitching_API
bool Stitcher_waveCorrection(
		struct StitcherPtr ptr);

extern "C" stitching_API
int Stitcher_waveCorrectKind(
		struct StitcherPtr ptr);

extern "C" stitching_API
double Stitcher_workScale(
		struct StitcherPtr ptr);