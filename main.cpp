#include <opencv4/opencv2/opencv.hpp>
#include <iostream>

// Parametreler
const cv::Scalar LOWER_FIRE_COLOR(0, 50, 100);    // Kırmızı rengin HSV alt sınırı (170°)
const cv::Scalar UPPER_FIRE_COLOR(255, 255, 255); // Kırmızı rengin HSV üst sınırı (180°)
const double FREQUENCY_THRESHOLD = 5.5;           // Frekans eşiği
const int PATCH_SIZE = 32;                        // Patch boyutu

double computeFrequency(const cv::Mat &patch)
{
    cv::Mat gray, complexI, magnitude;

    // Renkli görüntüyü gri tonlamaya dönüştür
    if (patch.channels() == 3)
    {
        cv::cvtColor(patch, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = patch.clone();
    }

    // Gri tonlama görüntüyü uygun boyutlara pad et
    int m = cv::getOptimalDFTSize(gray.rows);
    int n = cv::getOptimalDFTSize(gray.cols);
    cv::Mat padded;
    cv::copyMakeBorder(gray, padded, 0, m - gray.rows, 0, n - gray.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // Kompleks planı oluştur
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::merge(planes, 2, complexI);

    // DFT uygulaması
    cv::dft(complexI, complexI);

    // Frekans bileşenlerinin büyüklüğünü hesapla
    cv::split(complexI, planes);
    cv::magnitude(planes[0], planes[1], magnitude);
    magnitude += cv::Scalar::all(1); // Logaritmik ölçekleme için +1 ekleyin
    cv::log(magnitude, magnitude);

    // Büyüklük matrisinin merkezini kaydır
    magnitude = magnitude(cv::Rect(0, 0, magnitude.cols & -2, magnitude.rows & -2));
    int cx = magnitude.cols / 2;
    int cy = magnitude.rows / 2;
    cv::Mat q0(magnitude, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(magnitude, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(magnitude, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(magnitude, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    // Frekans değerlerinin ortalamasını hesapla
    cv::Scalar meanFreq = cv::mean(magnitude);
    return meanFreq[0];
}

int main()
{
    cv::VideoCapture cap(0); // Kamera aç

    if (!cap.isOpened())
    {
        std::cerr << "Kamera açılamadı!" << std::endl;
        return -1;
    }

    cv::Mat frame, hsvFrame, mask, patch, grayFrame, prevGrayFrame, diff;
    bool firstFrame = true;

    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;

        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        if (firstFrame)
        {
            prevGrayFrame = grayFrame.clone();
            firstFrame = false;
            continue;
        }

        // Hareket algılama
        cv::absdiff(grayFrame, prevGrayFrame, diff);
        prevGrayFrame = grayFrame.clone();

        // Ateş rengi modellemesi ve maskeleme
        cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);
        cv::inRange(hsvFrame, LOWER_FIRE_COLOR, UPPER_FIRE_COLOR, mask);

        // Maskeyi hareketle birleştir
        cv::bitwise_and(mask, diff, mask);

        // // Patch-wise işlem
        for (int y = 0; y < frame.rows; y += PATCH_SIZE)
        {
            for (int x = 0; x < frame.cols; x += PATCH_SIZE)
            {
                cv::Rect roi(x, y, PATCH_SIZE, PATCH_SIZE);
                roi &= cv::Rect(0, 0, frame.cols, frame.rows);
                patch = mask(roi);

                if (patch.rows > 0 && patch.cols > 0)
                {
                    double frequency = computeFrequency(patch);

                    // Frekans eşik kontrolü
                    if (frequency > FREQUENCY_THRESHOLD)
                    {
                        cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2); // Ateşi yeşil dikdörtgenle işaretle
                    }
                }
            }
        }

        cv::imshow("Fire Detection", frame);

        if (cv::waitKey(30) >= 0)
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
