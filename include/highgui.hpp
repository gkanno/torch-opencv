#include <Common.hpp>
#include <opencv2/highgui.hpp>

// highgui_EXPORTS is defined by CMake(add_library)
#if defined (_WIN32) && defined (_MSC_VER)
  #if defined(highgui_EXPORTS)
    #define  highgui_API __declspec(dllexport)
  #else
    #define  highgui_API __declspec(dllimport)
  #endif /* highgui_EXPORTS */
#else /* defined (_WIN32) */
 #define highgui_API
#endif

extern "C" highgui_API
void imshow(const char *winname, struct TensorWrapper mat);

extern "C" highgui_API
int waitKey(int delay);

extern "C" highgui_API
void namedWindow(const char *winname, int flags);

extern "C" highgui_API
void destroyWindow(const char *winname);

extern "C" highgui_API
void destroyAllWindows();

extern "C" highgui_API
int startWindowThread();

extern "C" highgui_API
void resizeWindow(const char *winname, int width, int height);

extern "C" highgui_API
void moveWindow(const char *winname, int x, int y);

extern "C" highgui_API
void setWindowProperty(const char *winname, int prop_id, double prop_value);

extern "C" highgui_API
void setWindowTitle(const char *winname, const char *title);

extern "C" highgui_API
double getWindowProperty(const char *winname, int prop_id);

extern "C" highgui_API
void setMouseCallback(const char *winname, cv::MouseCallback onMouse, void *userdata);

extern "C" highgui_API
int getMouseWheelData(int flags);

extern "C" highgui_API
int createTrackbar(
        const char *trackbarname, const char *winname, int *value,
        int count, cv::TrackbarCallback onChange, void *userdata);

extern "C" highgui_API
int getTrackbarPos(const char *trackbarname, const char *winname);

extern "C" highgui_API
void setTrackbarPos(const char *trackbarname, const char *winname, int pos);

extern "C" highgui_API
void setTrackbarMax(const char *trackbarname, const char *winname, int maxval);

extern "C" highgui_API
void updateWindow(const char *winname);

extern "C" highgui_API
void displayOverlay(const char *winname, const char *text, int delayms);

extern "C" highgui_API
void displayStatusBar(const char *winname, const char *text, int delayms);

extern "C" highgui_API
void saveWindowParameters(const char *windowName);

extern "C" highgui_API
void loadWindowParameters(const char *windowName);