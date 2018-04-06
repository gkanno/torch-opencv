#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/ml.hpp>

// ml_EXPORTS is defined by CMake(add_library)
#if defined (_WIN32) && defined (_MSC_VER)
  #if defined(ml_EXPORTS)
    #define  ml_API __declspec(dllexport)
  #else
    #define  ml_API __declspec(dllimport)
  #endif /* ml_EXPORTS */
#else /* defined (_WIN32) */
 #define ml_API
#endif

namespace ml = cv::ml;

struct TensorWrapper randMVNormal(
        struct TensorWrapper mean, struct TensorWrapper cov, int nsamples, struct TensorWrapper samples);

struct TensorArray createConcentricSpheresTestSet(
        int nsamples, int nfeatures, int nclasses, struct TensorWrapper samples, struct TensorWrapper responses);

struct ParamGridPtr {
    void *ptr;

    inline ml::ParamGrid * operator->() { return static_cast<ml::ParamGrid *>(ptr); }
    inline ParamGridPtr(ml::ParamGrid *ptr) { this->ptr = ptr; }

    inline operator ml::ParamGrid & () { return *static_cast<ml::ParamGrid *>(ptr); }
};

struct TrainDataPtr {
    void *ptr;

    inline ml::TrainData * operator->() { return static_cast<ml::TrainData *>(ptr); }
    inline TrainDataPtr(ml::TrainData *ptr) { this->ptr = ptr; }
    inline operator ml::TrainData *() { return static_cast<ml::TrainData *>(ptr); }
};

struct StatModelPtr {
    void *ptr;

    inline ml::StatModel * operator->() { return static_cast<ml::StatModel *>(ptr); }
    inline StatModelPtr(ml::StatModel *ptr) { this->ptr = ptr; }
};

extern "C" ml_API
struct ParamGridPtr ParamGrid_ctor(double _minVal, double _maxVal, double _logStep);

extern "C" ml_API
struct ParamGridPtr ParamGrid_ctor_default();

extern "C" ml_API
void ParamGrid_dtor(struct ParamGridPtr ptr);

extern "C" ml_API
struct TrainDataPtr TrainData_ctor(
        struct TensorWrapper samples, int layout, struct TensorWrapper responses,
        struct TensorWrapper varIdx, struct TensorWrapper sampleIdx,
        struct TensorWrapper sampleWeights, struct TensorWrapper varType);

extern "C" ml_API
void TrainData_dtor(struct TrainDataPtr ptr);

extern "C" ml_API
struct TensorWrapper TrainData_getSubVector(struct TensorWrapper vec, struct TensorWrapper idx);

extern "C" ml_API
int TrainData_getLayout(struct TrainDataPtr ptr);

extern "C" ml_API
int TrainData_getNTrainSamples(struct TrainDataPtr ptr);

extern "C" ml_API
int TrainData_getNTestSamples(struct TrainDataPtr ptr);

extern "C" ml_API
int TrainData_getNSamples(struct TrainDataPtr ptr);

extern "C" ml_API
int TrainData_getNVars(struct TrainDataPtr ptr);

extern "C" ml_API
int TrainData_getNAllVars(struct TrainDataPtr ptr);

extern "C" ml_API
struct TensorWrapper TrainData_getSamples(struct TrainDataPtr ptr);

extern "C" ml_API
struct TensorWrapper TrainData_getMissing(struct TrainDataPtr ptr);

extern "C" ml_API
struct TensorWrapper TrainData_getTrainResponses(struct TrainDataPtr ptr);

extern "C" ml_API
struct TensorWrapper TrainData_getTrainNormCatResponses(struct TrainDataPtr ptr);

extern "C" ml_API
struct TensorWrapper TrainData_getTestResponses(struct TrainDataPtr ptr);

extern "C" ml_API
struct TensorWrapper TrainData_getTestNormCatResponses(struct TrainDataPtr ptr);

extern "C" ml_API
struct TensorWrapper TrainData_getResponses(struct TrainDataPtr ptr);

extern "C" ml_API
struct TensorWrapper TrainData_getNormCatResponses(struct TrainDataPtr ptr);

extern "C" ml_API
struct TensorWrapper TrainData_getSampleWeights(struct TrainDataPtr ptr);

extern "C" ml_API
struct TensorWrapper TrainData_getTrainSampleWeights(struct TrainDataPtr ptr);

extern "C" ml_API
struct TensorWrapper TrainData_getTestSampleWeights(struct TrainDataPtr ptr);

extern "C" ml_API
struct TensorWrapper TrainData_getVarIdx(struct TrainDataPtr ptr);

extern "C" ml_API
struct TensorWrapper TrainData_getVarType(struct TrainDataPtr ptr);

extern "C" ml_API
int TrainData_getResponseType(struct TrainDataPtr ptr);

extern "C" ml_API
struct TensorWrapper TrainData_getTrainSampleIdx(struct TrainDataPtr ptr);

extern "C" ml_API
struct TensorWrapper TrainData_getTestSampleIdx(struct TrainDataPtr ptr);

extern "C" ml_API
struct TensorWrapper TrainData_getDefaultSubstValues(struct TrainDataPtr ptr);

extern "C" ml_API
struct TensorWrapper TrainData_getClassLabels(struct TrainDataPtr ptr);

extern "C" ml_API
struct TensorWrapper TrainData_getCatOfs(struct TrainDataPtr ptr);

extern "C" ml_API
struct TensorWrapper TrainData_getCatMap(struct TrainDataPtr ptr);

extern "C" ml_API
void TrainData_shuffleTrainTest(struct TrainDataPtr ptr);

extern "C" ml_API
struct TensorWrapper TrainData_getSample(
        struct TrainDataPtr ptr, struct TensorWrapper varIdx, int sidx);

extern "C" ml_API
struct TensorWrapper TrainData_getTrainSamples(
        struct TrainDataPtr ptr, int layout, bool compressSamples, bool compressVars);

extern "C" ml_API
struct TensorWrapper TrainData_getValues(
        struct TrainDataPtr ptr, int vi, struct TensorWrapper sidx);

extern "C" ml_API
struct TensorWrapper TrainData_getNormCatValues(
        struct TrainDataPtr ptr, int vi, struct TensorWrapper sidx);

extern "C" ml_API
void TrainData_setTrainTestSplit(struct TrainDataPtr ptr, int count, bool shuffle);

extern "C" ml_API
void TrainData_setTrainTestSplitRatio(struct TrainDataPtr ptr, double ratio, bool shuffle);

extern "C" ml_API
int StatModel_getVarCount(struct StatModelPtr ptr);

extern "C" ml_API
bool StatModel_empty(struct StatModelPtr ptr);

extern "C" ml_API
bool StatModel_isTrained(struct StatModelPtr ptr);

extern "C" ml_API
bool StatModel_isClassifier(struct StatModelPtr ptr);

extern "C" ml_API
bool StatModel_train(struct StatModelPtr ptr, struct TrainDataPtr trainData, int flags);

extern "C" ml_API
bool StatModel_train_Mat(
        struct StatModelPtr ptr, struct TensorWrapper samples, int layout, struct TensorWrapper responses);

extern "C" ml_API
struct TensorPlusFloat StatModel_calcError(
        struct StatModelPtr ptr, struct TrainDataPtr data, bool test, struct TensorWrapper resp);

extern "C" ml_API
float StatModel_predict(
        struct StatModelPtr ptr, struct TensorWrapper samples, struct TensorWrapper results, int flags);

struct NormalBayesClassifierPtr {
    void *ptr;

    inline ml::NormalBayesClassifier * operator->() { return static_cast<ml::NormalBayesClassifier *>(ptr); }
    inline NormalBayesClassifierPtr(ml::NormalBayesClassifier *ptr) { this->ptr = ptr; }
};

extern "C" ml_API
struct NormalBayesClassifierPtr NormalBayesClassifier_load(const char *filename, const char *objname);

extern "C" ml_API
struct NormalBayesClassifierPtr NormalBayesClassifier_ctor();

extern "C" ml_API
struct TensorArrayPlusFloat NormalBayesClassifier_predictProb(
        struct NormalBayesClassifierPtr ptr, struct TensorWrapper inputs,
        struct TensorWrapper outputs, struct TensorWrapper outputProbs, int flags);

struct KNearestPtr {
    void *ptr;

    inline ml::KNearest * operator->() { return static_cast<ml::KNearest *>(ptr); }
    inline KNearestPtr(ml::KNearest *ptr) { this->ptr = ptr; }
};

extern "C" ml_API
struct KNearestPtr KNearest_load(const char *filename, const char *objname);

extern "C" ml_API
struct KNearestPtr KNearest_ctor();

extern "C" ml_API
void KNearest_setDefaultK(struct KNearestPtr ptr, int val);

extern "C" ml_API
int KNearest_getDefaultK(struct KNearestPtr ptr);

extern "C" ml_API
void KNearest_setIsClassifier(struct KNearestPtr ptr, bool val);

extern "C" ml_API
bool KNearest_getIsClassifier(struct KNearestPtr ptr);

extern "C" ml_API
void KNearest_setEmax(struct KNearestPtr ptr, int val);

extern "C" ml_API
int KNearest_getEmax(struct KNearestPtr ptr);

extern "C" ml_API
void KNearest_setAlgorithmType(struct KNearestPtr ptr, int val);

extern "C" ml_API
int KNearest_getAlgorithmType(struct KNearestPtr ptr);

extern "C" ml_API
float KNearest_findNearest(
        struct KNearestPtr ptr, struct TensorWrapper samples, int k,
        struct TensorWrapper results, struct TensorWrapper neighborResponses,
        struct TensorWrapper dist);

struct SVMPtr {
    void *ptr;

    inline ml::SVM * operator->() { return static_cast<ml::SVM *>(ptr); }
    inline SVMPtr(ml::SVM *ptr) { this->ptr = ptr; }
};

extern "C" ml_API
struct SVMPtr SVM_load(const char *filename, const char *objname);

extern "C" ml_API
struct SVMPtr SVM_ctor();

extern "C" ml_API
void SVM_setType(struct SVMPtr ptr, int val);

extern "C" ml_API
int SVM_getType(struct SVMPtr ptr);

extern "C" ml_API
void SVM_setGamma(struct SVMPtr ptr, double val);

extern "C" ml_API
double SVM_getGamma(struct SVMPtr ptr);

extern "C" ml_API
void SVM_setCoef0(struct SVMPtr ptr, double val);

extern "C" ml_API
double SVM_getCoef0(struct SVMPtr ptr);

extern "C" ml_API
void SVM_setDegree(struct SVMPtr ptr, double val);

extern "C" ml_API
double SVM_getDegree(struct SVMPtr ptr);

extern "C" ml_API
void SVM_setC(struct SVMPtr ptr, double val);

extern "C" ml_API
double SVM_getC(struct SVMPtr ptr);

extern "C" ml_API
void SVM_setNu(struct SVMPtr ptr, double val);

extern "C" ml_API
double SVM_getNu(struct SVMPtr ptr);

extern "C" ml_API
void SVM_setP(struct SVMPtr ptr, double val);

extern "C" ml_API
double SVM_getP(struct SVMPtr ptr);

extern "C" ml_API
void SVM_setClassWeights(struct SVMPtr ptr, struct TensorWrapper val);

extern "C" ml_API
struct TensorWrapper SVM_getClassWeights(struct SVMPtr ptr);

extern "C" ml_API
void SVM_setTermCriteria(struct SVMPtr ptr, struct TermCriteriaWrapper val);

extern "C" ml_API
struct TermCriteriaWrapper SVM_getTermCriteria(struct SVMPtr ptr);

extern "C" ml_API
int SVM_getKernelType(struct SVMPtr ptr);

extern "C" ml_API
void SVM_setKernel(struct SVMPtr ptr, int val);

//extern "C" ml_API
//void SVM_setCustomKernel(struct SVMPtr ptr, struct KernelPtr val);

extern "C" ml_API
bool SVM_trainAuto(
        struct SVMPtr ptr, struct TrainDataPtr data, int kFold, struct ParamGridPtr Cgrid,
        struct ParamGridPtr gammaGrid, struct ParamGridPtr pGrid, struct ParamGridPtr nuGrid,
        struct ParamGridPtr coeffGrid, struct ParamGridPtr degreeGrid, bool balanced);

extern "C" ml_API
struct TensorWrapper SVM_getSupportVectors(struct SVMPtr ptr);

extern "C" ml_API
struct TensorArrayPlusDouble SVM_getDecisionFunction(
        struct SVMPtr ptr, int i, struct TensorWrapper alpha, struct TensorWrapper svidx);

extern "C" ml_API
struct ParamGridPtr SVM_getDefaultGrid(int param_id);

struct EMPtr {
    void *ptr;

    inline ml::EM * operator->() { return static_cast<ml::EM *>(ptr); }
    inline EMPtr(ml::EM *ptr) { this->ptr = ptr; }
};

extern "C" ml_API
struct EMPtr EM_load(const char *filename, const char *objname);

extern "C" ml_API
struct EMPtr EM_ctor();

extern "C" ml_API
void EM_setClustersNumber(struct EMPtr ptr, int val);

extern "C" ml_API
int EM_getClustersNumber(struct EMPtr ptr);

extern "C" ml_API
void EM_setCovarianceMatrixType(struct EMPtr ptr, int val);

extern "C" ml_API
int EM_getCovarianceMatrixType(struct EMPtr ptr);

extern "C" ml_API
void EM_setTermCriteria(struct EMPtr ptr, struct TermCriteriaWrapper val);

extern "C" ml_API
struct TermCriteriaWrapper EM_getTermCriteria(struct EMPtr ptr);

extern "C" ml_API
struct TensorWrapper EM_getWeights(struct EMPtr ptr);

extern "C" ml_API
struct TensorWrapper EM_getMeans(struct EMPtr ptr);

extern "C" ml_API
struct TensorArray EM_getCovs(struct EMPtr ptr);

extern "C" ml_API
struct Vec2dWrapper EM_predict2(
        struct EMPtr ptr, struct TensorWrapper sample, struct TensorWrapper probs);

extern "C" ml_API
bool EM_trainEM(
        struct EMPtr ptr, struct TensorWrapper samples,
        struct TensorWrapper logLikelihoods,
        struct TensorWrapper labels, struct TensorWrapper probs);

extern "C" ml_API
bool EM_trainE(
        struct EMPtr ptr, struct TensorWrapper samples, struct TensorWrapper means0,
        struct TensorWrapper covs0, struct TensorWrapper weights0,
        struct TensorWrapper logLikelihoods, struct TensorWrapper labels,
        struct TensorWrapper probs);

extern "C" ml_API
bool EM_trainM(
        struct EMPtr ptr, struct TensorWrapper samples, struct TensorWrapper probs0,
        struct TensorWrapper logLikelihoods, struct TensorWrapper labels,
        struct TensorWrapper probs);

struct DTreesPtr {
    void *ptr;

    inline ml::DTrees * operator->() { return static_cast<ml::DTrees *>(ptr); }
    inline DTreesPtr(ml::DTrees *ptr) { this->ptr = ptr; }
};

struct ConstNodeArray {
    const ml::DTrees::Node *ptr;
    int size;
};

struct ConstSplitArray {
    const ml::DTrees::Split *ptr;
    int size;
};

extern "C" ml_API
struct DTreesPtr DTrees_ctor();

extern "C" ml_API
struct DTreesPtr DTrees_load(const char *filename, const char *objname);

extern "C" ml_API
void DTrees_setMaxCategories(struct DTreesPtr ptr, int val);

extern "C" ml_API
int DTrees_getMaxCategories(struct DTreesPtr ptr);

extern "C" ml_API
void DTrees_setMaxDepth(struct DTreesPtr ptr, int val);

extern "C" ml_API
int DTrees_getMaxDepth(struct DTreesPtr ptr);

extern "C" ml_API
void DTrees_setMinSampleCount(struct DTreesPtr ptr, int val);

extern "C" ml_API
int DTrees_getMinSampleCount(struct DTreesPtr ptr);

extern "C" ml_API
void DTrees_setCVFolds(struct DTreesPtr ptr, int val);

extern "C" ml_API
int DTrees_getCVFolds(struct DTreesPtr ptr);

extern "C" ml_API
void DTrees_setUseSurrogates(struct DTreesPtr ptr, bool val);

extern "C" ml_API
bool DTrees_getUseSurrogates(struct DTreesPtr ptr);

extern "C" ml_API
void DTrees_setUse1SERule(struct DTreesPtr ptr, bool val);

extern "C" ml_API
bool DTrees_getUse1SERule(struct DTreesPtr ptr);

extern "C" ml_API
void DTrees_setTruncatePrunedTree(struct DTreesPtr ptr, bool val);

extern "C" ml_API
bool DTrees_getTruncatePrunedTree(struct DTreesPtr ptr);

extern "C" ml_API
void DTrees_setRegressionAccuracy(struct DTreesPtr ptr, float val);

extern "C" ml_API
float DTrees_getRegressionAccuracy(struct DTreesPtr ptr);

extern "C" ml_API
void DTrees_setPriors(struct DTreesPtr ptr, struct TensorWrapper val);

extern "C" ml_API
struct TensorWrapper DTrees_getPriors(struct DTreesPtr ptr);

extern "C" ml_API
struct TensorWrapper DTrees_getRoots(struct DTreesPtr ptr);

extern "C" ml_API
struct ConstNodeArray DTrees_getNodes(struct DTreesPtr ptr);

extern "C" ml_API
struct ConstSplitArray DTrees_getSplits(struct DTreesPtr ptr);

extern "C" ml_API
struct TensorWrapper DTrees_getSubsets(struct DTreesPtr ptr);

struct RTreesPtr {
    void *ptr;

    inline ml::RTrees * operator->() { return static_cast<ml::RTrees *>(ptr); }
    inline RTreesPtr(ml::RTrees *ptr) { this->ptr = ptr; }
};

extern "C" ml_API
struct RTreesPtr RTrees_load(const char *filename, const char *objname);

extern "C" ml_API
struct RTreesPtr RTrees_ctor();

extern "C" ml_API
void RTrees_setCalculateVarImportance(struct RTreesPtr ptr, bool val);

extern "C" ml_API
bool RTrees_getCalculateVarImportance(struct RTreesPtr ptr);

extern "C" ml_API
void RTrees_setActiveVarCount(struct RTreesPtr ptr, int val);

extern "C" ml_API
int RTrees_getActiveVarCount(struct RTreesPtr ptr);

extern "C" ml_API
void RTrees_setTermCriteria(struct RTreesPtr ptr, struct TermCriteriaWrapper val);

struct BoostPtr {
    void *ptr;

    inline ml::Boost * operator->() { return static_cast<ml::Boost *>(ptr); }
    inline BoostPtr(ml::Boost *ptr) { this->ptr = ptr; }
};

extern "C" ml_API
struct BoostPtr Boost_load(const char *filename, const char *objname);

extern "C" ml_API
struct BoostPtr Boost_ctor();

extern "C" ml_API
void Boost_setBoostType(struct BoostPtr ptr, int val);

extern "C" ml_API
int Boost_getBoostType(struct BoostPtr ptr);

extern "C" ml_API
void Boost_setWeakCount(struct BoostPtr ptr, int val);

extern "C" ml_API
int Boost_getWeakCount(struct BoostPtr ptr);

extern "C" ml_API
void Boost_setWeightTrimRate(struct BoostPtr ptr, double val);

extern "C" ml_API
double Boost_getWeightTrimRate(struct BoostPtr ptr);

struct ANN_MLPPtr {
    void *ptr;

    inline ml::ANN_MLP * operator->() { return static_cast<ml::ANN_MLP *>(ptr); }
    inline ANN_MLPPtr(ml::ANN_MLP *ptr) { this->ptr = ptr; }
};

extern "C" ml_API
struct ANN_MLPPtr ANN_MLP_load(const char *filename, const char *objname);

extern "C" ml_API
struct ANN_MLPPtr ANN_MLP_ctor();

extern "C" ml_API
void ANN_MLP_setTrainMethod(struct ANN_MLPPtr ptr, int method, double param1, double param2);

extern "C" ml_API
int ANN_MLP_getTrainMethod(struct ANN_MLPPtr ptr);

extern "C" ml_API
void ANN_MLP_setActivationFunction(struct ANN_MLPPtr ptr, int type, double param1, double param2);

extern "C" ml_API
void ANN_MLP_setLayerSizes(struct ANN_MLPPtr ptr, struct TensorWrapper val);

extern "C" ml_API
struct TensorWrapper ANN_MLP_getLayerSizes(struct ANN_MLPPtr ptr);

extern "C" ml_API
void ANN_MLP_setTermCriteria(struct ANN_MLPPtr ptr, struct TermCriteriaWrapper val);

extern "C" ml_API
struct TermCriteriaWrapper ANN_MLP_getTermCriteria(struct ANN_MLPPtr ptr);

extern "C" ml_API
void ANN_MLP_setBackpropWeightScale(struct ANN_MLPPtr ptr, double val);

extern "C" ml_API
double ANN_MLP_getBackpropWeightScale(struct ANN_MLPPtr ptr);

extern "C" ml_API
void ANN_MLP_setBackpropMomentumScale(struct ANN_MLPPtr ptr, double val);

extern "C" ml_API
double ANN_MLP_getBackpropMomentumScale(struct ANN_MLPPtr ptr);

extern "C" ml_API
void ANN_MLP_setRpropDW0(struct ANN_MLPPtr ptr, double val);

extern "C" ml_API
double ANN_MLP_getRpropDW0(struct ANN_MLPPtr ptr);

extern "C" ml_API
void ANN_MLP_setRpropDWPlus(struct ANN_MLPPtr ptr, double val);

extern "C" ml_API
double ANN_MLP_getRpropDWPlus(struct ANN_MLPPtr ptr);

extern "C" ml_API
void ANN_MLP_setRpropDWMinus(struct ANN_MLPPtr ptr, double val);

extern "C" ml_API
double ANN_MLP_getRpropDWMinus(struct ANN_MLPPtr ptr);

extern "C" ml_API
void ANN_MLP_setRpropDWMin(struct ANN_MLPPtr ptr, double val);

extern "C" ml_API
double ANN_MLP_getRpropDWMin(struct ANN_MLPPtr ptr);

extern "C" ml_API
void ANN_MLP_setRpropDWMax(struct ANN_MLPPtr ptr, double val);

extern "C" ml_API
double ANN_MLP_getRpropDWMax(struct ANN_MLPPtr ptr);

extern "C" ml_API
struct TensorWrapper ANN_MLP_getWeights(struct ANN_MLPPtr ptr, int layerIdx);

struct LogisticRegressionPtr {
    void *ptr;

    inline ml::LogisticRegression * operator->() { return static_cast<ml::LogisticRegression *>(ptr); }
    inline LogisticRegressionPtr(ml::LogisticRegression *ptr) { this->ptr = ptr; }
};

extern "C" ml_API
struct LogisticRegressionPtr LogisticRegression_load(const char *filename, const char *objname);

extern "C" ml_API
struct LogisticRegressionPtr LogisticRegression_ctor();

extern "C" ml_API
void LogisticRegression_setLearningRate(struct LogisticRegressionPtr ptr, double val);

extern "C" ml_API
double LogisticRegression_getLearningRate(struct LogisticRegressionPtr ptr);

extern "C" ml_API
void LogisticRegression_setIterations(struct LogisticRegressionPtr ptr, int val);

extern "C" ml_API
int LogisticRegression_getIterations(struct LogisticRegressionPtr ptr);

extern "C" ml_API
void LogisticRegression_setRegularization(struct LogisticRegressionPtr ptr, int val);

extern "C" ml_API
int LogisticRegression_getRegularization(struct LogisticRegressionPtr ptr);

extern "C" ml_API
void LogisticRegression_setTrainMethod(struct LogisticRegressionPtr ptr, int val);

extern "C" ml_API
int LogisticRegression_getTrainMethod(struct LogisticRegressionPtr ptr);

extern "C" ml_API
void LogisticRegression_setMiniBatchSize(struct LogisticRegressionPtr ptr, int val);

extern "C" ml_API
int LogisticRegression_getMiniBatchSize(struct LogisticRegressionPtr ptr);

extern "C" ml_API
void LogisticRegression_setTermCriteria(struct LogisticRegressionPtr ptr, struct TermCriteriaWrapper val);

extern "C" ml_API
struct TermCriteriaWrapper LogisticRegression_getTermCriteria(struct LogisticRegressionPtr ptr);

extern "C" ml_API
struct TensorWrapper LogisticRegression_get_learnt_thetas(struct LogisticRegressionPtr ptr);
