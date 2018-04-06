#pragma once
#include <Common.hpp>


// Classes_EXPORTS is defined by CMake(add_library)
#if defined (_WIN32) && defined (_MSC_VER)
  #if defined(Classes_EXPORTS)
    #define  Classes_API __declspec(dllexport)
  #else
    #define  Classes_API __declspec(dllimport)
  #endif /* Classes_EXPORTS */
#else /* defined (_WIN32) */
 #define Classes_API
#endif

// This is cv::Ptr with all fields made public
// TODO I hope a safer solution to be here one day
template <typename T>
struct PublicPtr
{
public:
    cv::detail::PtrOwner* owner;
    T* stored;
};

// increfs a cv::Ptr and returns the pointer
template <typename T>
T *rescueObjectFromPtr(cv::Ptr<T> ptr) {
    PublicPtr<T> *publicPtr = reinterpret_cast<PublicPtr<T> *>(&ptr);
    publicPtr->owner->incRef();
    return ptr.get();
}

// FileNode

struct FileNodePtr {
    void *ptr;

    inline cv::FileNode * operator->() { return static_cast<cv::FileNode *>(ptr); }
    inline FileNodePtr(cv::FileNode *ptr) { this->ptr = ptr; }
    inline cv::FileNode & operator*() { return *static_cast<cv::FileNode *>(this->ptr); }
};

extern "C" Classes_API
struct FileNodePtr FileNode_ctor();

extern "C" Classes_API
void FileNode_dtor(FileNodePtr ptr);

// FileStorage

struct FileStoragePtr {
    void *ptr;

    inline cv::FileStorage * operator->() { return static_cast<cv::FileStorage *>(ptr); }
    inline FileStoragePtr(cv::FileStorage *ptr) { this->ptr = ptr; }
    inline cv::FileStorage & operator*() { return *static_cast<cv::FileStorage *>(this->ptr); }
};

extern "C" Classes_API
struct FileStoragePtr FileStorage_ctor_default();

extern "C" Classes_API
struct FileStoragePtr FileStorage_ctor(const char *source, int flags, const char *encoding);

extern "C" Classes_API
void FileStorage_dtor(FileStoragePtr ptr);

extern "C" Classes_API
bool FileStorage_open(FileStoragePtr ptr, const char *filename, int flags, const char *encoding);

extern "C" Classes_API
bool FileStorage_isOpened(FileStoragePtr ptr);

extern "C" Classes_API
void FileStorage_release(FileStoragePtr ptr);

extern "C" Classes_API
const char *FileStorage_releaseAndGetString(FileStoragePtr ptr);

// Algorithm

struct AlgorithmPtr {
    void *ptr;

    inline cv::Algorithm * operator->() { return static_cast<cv::Algorithm *>(ptr); }
    inline AlgorithmPtr(cv::Algorithm *ptr) { this->ptr = ptr; }
};

extern "C" Classes_API
struct AlgorithmPtr Algorithm_ctor();

extern "C" Classes_API
void Algorithm_dtor(AlgorithmPtr ptr);

extern "C" Classes_API
void Algorithm_clear(AlgorithmPtr ptr);

extern "C" Classes_API
void Algorithm_write(AlgorithmPtr ptr, FileStoragePtr fileStorage);

extern "C" Classes_API
void Algorithm_read(AlgorithmPtr ptr, FileNodePtr fileNode);

extern "C" Classes_API
bool Algorithm_empty(AlgorithmPtr ptr);

extern "C" Classes_API
void Algorithm_save(AlgorithmPtr ptr, const char *filename);

extern "C" Classes_API
const char *Algorithm_getDefaultName(AlgorithmPtr ptr);

// cv::Ptr

struct CvPtrPtr {
    void *ptr;

    inline cv::Ptr<void> * operator->() { return static_cast<cv::Ptr<void> *>(ptr); }
    inline CvPtrPtr(cv::Ptr<void> *ptr) { this->ptr = ptr; }
    inline cv::Ptr<void> & operator*() { return *static_cast<cv::Ptr<void> *>(this->ptr); }
};

extern "C" Classes_API
void CvPtr_dtor(struct CvPtrPtr ptr);

