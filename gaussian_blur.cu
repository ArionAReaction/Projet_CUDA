#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;
using namespace cv;

__global__ void gaussian_blur(unsigned char *image, unsigned char *image_out, double *kernel, int kernelSize, size_t cols, size_t rows ) {
    auto i = blockIdx.y * blockDim.y + threadIdx.y;
    auto j = blockIdx.x * blockDim.x + threadIdx.x;
    int kernelRadius = kernelSize / 2;
    double sum[3] = {0.0, 0.0, 0.0};
    for (int m = -kernelRadius; m <= kernelRadius; ++m) {
        for (int n = -kernelRadius; n <= kernelRadius; ++n) {
            int rowIndex = i + m;
            int colIndex = j + n;
            if (rowIndex >= 0 && rowIndex < rows && colIndex >= 0 && colIndex < cols) {
                sum[0] += kernel[(m + kernelRadius) * kernelSize + n + kernelRadius] *
                    image[rowIndex * cols * 3 + colIndex * 3];
                sum[1] += kernel[(m + kernelRadius) * kernelSize + n + kernelRadius] *
                    image[rowIndex * cols * 3 + colIndex * 3 + 1];
                sum[2] += kernel[(m + kernelRadius) * kernelSize + n + kernelRadius] *
                    image[rowIndex * cols * 3 + colIndex * 3 + 2];
            }
        }
    }
    image_out[i * cols * 3 + j * 3] = sum[0];
    image_out[i * cols * 3 + j * 3 + 1] = sum[1];
    image_out[i * cols * 3 + j * 3 + 2] = sum[2];
}

int main(int argc, char *argv[]){
    cudaEvent_t event_deb;
	cudaEvent_t event_fin;
	
	cudaError_t cudaErrorIdentifier;
	
	float temps;
	cudaErrorIdentifier = cudaEventCreate(&event_deb);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à la création de l'événement event_deb"<<endl;
		
	cudaErrorIdentifier = cudaEventCreate(&event_fin);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à la création de l'événement event_fin"<<endl;

	cudaErrorIdentifier = cudaEventRecord(event_deb,0);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur au lancement du record event_deb"<<endl;

    Mat image = imread(argv[1], IMREAD_UNCHANGED);
    int kernelSize = 5;
    double sigma = 5;

    Mat kernel(kernelSize, kernelSize, CV_64F);
    double mean = kernelSize / 2;
    double sum = 0.0;
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            kernel.at<double>(i, j) = exp(-0.5 * (pow((i - mean) / sigma, 2.0) + pow((j - mean) / sigma, 2.0)))
                                      / (2 * M_PI * sigma * sigma);
            sum += kernel.at<double>(i, j);
        }
    }
    kernel /= sum;
	
	cudaDeviceProp prop;
	cudaErrorIdentifier = cudaGetDeviceProperties(&prop, 0);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à la récupération des propriétés du device"<<endl;
		
    double *kernel_d;
    cudaErrorIdentifier = cudaMalloc(&kernel_d, kernelSize * kernelSize * sizeof(double));
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à l'allocation de kernel_d"<<endl;

    cudaErrorIdentifier = cudaMemcpy(kernel_d, kernel.data, kernelSize * kernelSize * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à l'envoi des données à kernel_d"<<endl;

    size_t const rows = image.rows;
    size_t const cols = image.cols;
    size_t const size = rows * cols;

    vector<unsigned char> blurred(size * 3);
    Mat image_out(rows, cols, CV_8UC3, blurred.data());
    unsigned char *image_d;
    unsigned char *image_out_d;
    
    cudaErrorIdentifier = cudaMalloc(&image_d, size * sizeof(unsigned char) * 3);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à l'allocation de image_d"<<endl;
		
    cudaErrorIdentifier = cudaMalloc(&image_out_d, size * sizeof(unsigned char) * 3);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à l'allocation de image_out_d"<<endl;
    
    cudaErrorIdentifier = cudaMemcpy(image_d, image.data, size * sizeof(unsigned char) * 3, cudaMemcpyHostToDevice);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à l'envoi des données à image_d"<<endl;

    dim3 block(sqrt(prop.maxThreadsPerBlock), sqrt(prop.maxThreadsPerBlock));
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    gaussian_blur<<<grid, block>>>( image_d, image_out_d, kernel_d, kernelSize, cols, rows );
	cudaErrorIdentifier = cudaGetLastError();
	if (cudaErrorIdentifier != cudaSuccess){
		cout<<"Erreur à l'exécution de la fonction gaussian_blur"<<endl;
		cout<<"détail : "<<cudaGetErrorString(cudaErrorIdentifier)<<endl;
	}
    cudaErrorIdentifier = cudaMemcpy(blurred.data(), image_out_d, size * sizeof(unsigned char) * 3, cudaMemcpyDeviceToHost);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à la réception des données de image_out_d"<<endl;

	std::string path(argv[1]);
	imwrite( "out_gaussian_blur_cu_"+path, image_out );

	cudaErrorIdentifier = cudaEventRecord(event_fin,0);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur au lancement du record event_fin"<<endl;
		
	cudaErrorIdentifier = cudaEventSynchronize(event_fin);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à la synchronisation entre event_deb et event_fin"<<endl;
		
	cudaErrorIdentifier = cudaEventElapsedTime(&temps,event_deb,event_fin);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur au calcul du temps entre event_deb et event_fin"<<endl;
		
	cout<<temps<<" millisecondes"<<endl;
	
	cudaErrorIdentifier = cudaEventDestroy(event_deb);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à la destruction de event_deb"<<endl;
		
	cudaErrorIdentifier = cudaEventDestroy(event_fin);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à la destruction de event_fin"<<endl;
    
    cudaFree(kernel_d);
    cudaFree(image_d);
    cudaFree(image_out_d);

    return 0;
}
