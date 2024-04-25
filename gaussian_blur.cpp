#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;
using namespace cv;
using namespace chrono;

int main(int argc, char *argv[]){
	auto start = high_resolution_clock::now();
	Mat image = imread(argv[1], IMREAD_UNCHANGED );

    int kernelSize = 5;
    double sigma = 5;
    
    Mat kernel(kernelSize, kernelSize, CV_64F);
    double mean = kernelSize / 2;
    double sum = 0.0;
    for (int x = 0; x < kernelSize; ++x) {
        for (int y = 0; y < kernelSize; ++y) {
            kernel.at<double>(x, y) = exp(-0.5 * (pow((x - mean) / sigma, 2.0) + pow((y - mean) / sigma, 2.0)))
                                      / (2 * M_PI * sigma * sigma);
            //kernel.at<double>(x, y) = exp(-0.5 * (pow(x / sigma, 2.0) + pow(y / sigma, 2.0)))
            //                          / (2 * M_PI * sigma * sigma);
            sum += kernel.at<double>(x, y);
        }
    }
    kernel /= sum;
    
    unsigned char *rgb = image.data;

    vector<unsigned char> blurred(image.rows * image.cols * 3);
    // Mat image_out(image.rows, image.cols, CV_8UC1, blurred.data());
    Mat image_out(image.rows, image.cols, CV_8UC3, blurred.data());
    int kernelRadius = kernel.rows / 2;
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            double sum[3] = {0.0, 0.0, 0.0};
            for (int m = -kernelRadius; m <= kernelRadius; ++m) {
                for (int n = -kernelRadius; n <= kernelRadius; ++n) {
                    int rowIndex = i + m;
                    int colIndex = j + n;
                    if (rowIndex >= 0 && rowIndex < image.rows &&
                        colIndex >= 0 && colIndex < image.cols) {
                        sum[0] += kernel.at<double>(m + kernelRadius, n + kernelRadius) *
                               rgb[rowIndex * image.cols * 3 + colIndex * 3];
                        sum[1] += kernel.at<double>(m + kernelRadius, n + kernelRadius) *
                               rgb[rowIndex * image.cols * 3 + colIndex * 3 + 1];
                        sum[2] += kernel.at<double>(m + kernelRadius, n + kernelRadius) *
                               rgb[rowIndex * image.cols * 3 + colIndex * 3 + 2];
                    }
                }
            }
            blurred[i * image.cols * 3 + j * 3] = sum[0];
            blurred[i * image.cols * 3 + j * 3 + 1] = sum[1];
            blurred[i * image.cols * 3 + j * 3 + 2] = sum[2];
        }
    }
	auto end = high_resolution_clock::now();

    cout << "Temps :" << endl;
    cout << "    " << duration_cast<seconds>(end - start).count() << " secondes" << endl;
    cout << "    " << duration_cast<milliseconds>(end - start).count() << " millisecondes" << endl;
    cout << "    " << duration_cast<microseconds>(end - start).count() << " microsecondes" << endl;

	std::string path(argv[1]);
	imwrite( "out_gaussian_blur_"+path, image_out );
    
    return 0;
}
