#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
using namespace std;
using namespace cv;

__global__ void grayscale(unsigned char* mA_inter,unsigned char* rgb,int width){
	auto i = blockIdx.y * blockDim.y + threadIdx.y;
	auto j = blockIdx.x * blockDim.x + threadIdx.x;
	mA_inter[i*width+j] = (307*rgb[3*(i*width+j)]+604*rgb[3*(i*width+j)+1]+113*rgb[3*(i*width+j)+2])/1024;
}

__global__ void sobel(unsigned char* mA_d,unsigned char* mA_inter,int width,int height){
	auto i = blockIdx.y * blockDim.y + threadIdx.y;
	auto j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > 0 && i < height && j % width != 0 && j % width != width-1){
		float h = mA_inter[(i-1)*width+(j-1)] - mA_inter[(i-1)*width+(j+1)]
				+ 2*mA_inter[i*width+(j-1)] - 2*mA_inter[i*width+(j+1)]
				+ mA_inter[(i+1)*width+(j-1)] - mA_inter[(i+1)*width+(j+1)];
		float v = mA_inter[(i-1)*width+(j-1)] - mA_inter[(i+1)*width+(j-1)]
				+ 2*mA_inter[(i-1)*width+j] - 2*mA_inter[(i+1)*width+j]
				+ mA_inter[(i-1)*width+(j+1)] - mA_inter[(i+1)*width+(j+1)];
		h = h > 255 ? 255 : h;
		v = v > 255 ? 255 : v;
		float res = h*h + v*v;
		res = res > 255*255 ? 255*255 : res;
		mA_d[i*width+j] = sqrt(res);
	}
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

	Mat m_in = imread(argv[1], IMREAD_UNCHANGED );
	auto rgb = m_in.data;
	
	vector< unsigned char > g( m_in.rows * m_in.cols);
	Mat m_out( m_in.rows, m_in.cols, CV_8UC1, g.data());
	
	unsigned char* mA_d=nullptr;
	unsigned char* mA_inter=nullptr;
	unsigned char* mA_rgb=nullptr;
	
	cudaDeviceProp prop;
	cudaErrorIdentifier = cudaGetDeviceProperties(&prop, 0);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à la récupération des propriétés du device"<<endl;
	
	cudaErrorIdentifier = cudaMalloc(&mA_d,g.size()*sizeof(unsigned char));
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à l'allocation de mA_d"<<endl;
	cudaErrorIdentifier = cudaMalloc(&mA_inter,g.size()*sizeof(unsigned char));
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à l'allocation de mA_inter"<<endl;
	cudaErrorIdentifier = cudaMalloc(&mA_rgb,m_in.rows*m_in.cols*3*sizeof(unsigned char));
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à l'allocation de mA_rgb"<<endl;
	
	cudaErrorIdentifier = cudaMemcpy(mA_d,g.data(),g.size()*sizeof(unsigned char),cudaMemcpyHostToDevice);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à l'envoi des données à mA_d"<<endl;
	cudaErrorIdentifier = cudaMemcpy(mA_inter,g.data(),g.size()*sizeof(unsigned char),cudaMemcpyHostToDevice);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à l'envoi des données à mA_inter"<<endl;
	cudaErrorIdentifier = cudaMemcpy(mA_rgb,rgb,m_in.rows*m_in.cols*3*sizeof(unsigned char),cudaMemcpyHostToDevice);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à l'envoi des données à mA_rgb"<<endl;
	
	/*int grid = 1;
	int block = m_in.rows * m_in.cols;
	while (block > prop.maxThreadsPerBlock){
		block -= prop.maxThreadsPerBlock;
		grid++;
	}
	if (grid > 1){
		block = prop.maxThreadsPerBlock;
	}*/
	dim3 block(sqrt(prop.maxThreadsPerBlock), sqrt(prop.maxThreadsPerBlock));
	dim3 grid((m_in.cols + block.x - 1) / block.x, (m_in.rows + block.y - 1) / block.y);
	grayscale<<<grid,block>>>(mA_inter,mA_rgb,m_in.cols);
	
	cudaErrorIdentifier = cudaGetLastError();
	if (cudaErrorIdentifier != cudaSuccess){
		cout<<"Erreur à l'exécution de la fonction grayscale"<<endl;
		cout<<"détail : "<<cudaGetErrorString(cudaErrorIdentifier)<<endl;
	}
	cudaErrorIdentifier = cudaMemcpy(g.data(),mA_inter,g.size()*sizeof(unsigned char),cudaMemcpyDeviceToHost);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à la réception des données de mA_inter"<<endl;

	cudaErrorIdentifier = cudaMemcpy(mA_inter,g.data(),g.size()*sizeof(unsigned char),cudaMemcpyHostToDevice);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à l'envoi des données à mA_inter"<<endl;

	sobel<<<grid,block>>>(mA_d,mA_inter,m_in.cols,m_in.rows);
	cudaErrorIdentifier = cudaGetLastError();
	if (cudaErrorIdentifier != cudaSuccess){
		cout<<"Erreur à l'exécution de la fonction sobel"<<endl;
		cout<<"détail : "<<cudaGetErrorString(cudaErrorIdentifier)<<endl;
	}
	cudaErrorIdentifier = cudaMemcpy(g.data(),mA_d,g.size()*sizeof(unsigned char),cudaMemcpyDeviceToHost);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à la réception des données de mA_d"<<endl;

	std::string path(argv[1]);
	imwrite( "out_sobel_cu_"+path, m_out );
	
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

	cudaFree(mA_d);
	cudaFree(mA_inter);
	cudaFree(mA_rgb);
	return 0;
}



