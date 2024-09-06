#include <opencv4/opencv2/opencv.hpp>
#include <iostream>

const cv::Scalar LOWER_RED_HSV(0, 50, 100);    // Kırmızı rengin HSV alt sınırı (170°)
const cv::Scalar UPPER_RED_HSV(255, 255, 255); // Kırmızı rengin HSV üst sınırı (180°)
const int PATCH_SIZE = 4;                      // Patch boyutu

int main()
{
    cv::VideoCapture cap(0); // Kamera aç

    if (!cap.isOpened())
    {
        std::cerr << "Kamera açılamadı!" << std::endl;
        return -1;
    }

    cv::Mat frame, hsvFrame, mask, patch;

    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;

        // Renk alanında kırmızı rengi algıla
        cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);
        cv::inRange(hsvFrame, LOWER_RED_HSV, UPPER_RED_HSV, mask);

        // Patch-wise işlemi
        for (int y = 0; y < frame.rows; y += PATCH_SIZE)
        {
            for (int x = 0; x < frame.cols; x += PATCH_SIZE)
            {
                cv::Rect roi(x, y, PATCH_SIZE, PATCH_SIZE);
                roi &= cv::Rect(0, 0, frame.cols, frame.rows); // Görüntü sınırlarına uyum sağla
                patch = mask(roi);

                if (cv::countNonZero(patch) > 0)
                {                                                        // Eğer patch içinde kırmızı varsa
                    cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2); // Yeşil dikdörtgenle işaretle
                }
            }
        }

        cv::imshow("Red Detection", frame);

        if (cv::waitKey(30) >= 0)
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
