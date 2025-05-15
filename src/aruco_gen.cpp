#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>

int main() {
    cv::aruco::GridBoard board(cv::Size(4, 6), 0.1, 0.02, cv::aruco::Dictionary());
}
