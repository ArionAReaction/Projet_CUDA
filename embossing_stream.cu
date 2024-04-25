#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <chrono>

using namespace std;
using namespace cv;

__global__ void embossing(unsigned char* mA_dest,unsigned char* rgb,int width,int height){
	auto i = blockIdx.y * blockDim.y + threadIdx.y;
	auto j = blockIdx.x * blockDim.x + threadIdx.x;
	int filter[] = {0,0,-1,0,0,0,1,0,0};
	if (i > 0 && i < height && j % width != 0 && j % width != width-1){
		mA_dest[i*width*3+j*3] =
			  rgb[3*((i-1)*width+(j-1))]*filter[0]
			+ rgb[3*((i-1)*width+j)]*filter[1]
			+ rgb[3*((i-1)*width+(j+1))]*filter[2]
			+ rgb[3*(i*width+(j-1))]*filter[3]
			+ rgb[3*(i*width+j)]*filter[4]
			+ rgb[3*(i*width+(j+1))]*filter[5]
			+ rgb[3*((i+1)*width+(j-1))]*filter[6]
			+ rgb[3*((i+1)*width+j)]*filter[7]
			+ rgb[3*((i+1)*width+(j+1))]*filter[8]+128;
		mA_dest[i*width*3+j*3+1] =
			  rgb[3*((i-1)*width+(j-1))+1]*filter[0]
			+ rgb[3*((i-1)*width+j)+1]*filter[1]
			+ rgb[3*((i-1)*width+(j+1))+1]*filter[2]
			+ rgb[3*(i*width+(j-1))+1]*filter[3]
			+ rgb[3*(i*width+j)+1]*filter[4]
			+ rgb[3*(i*width+(j+1))+1]*filter[5]
			+ rgb[3*((i+1)*width+(j-1))+1]*filter[6]
			+ rgb[3*((i+1)*width+j)+1]*filter[7]
			+ rgb[3*((i+1)*width+(j+1))+1]*filter[8]+128;
		mA_dest[i*width*3+j*3+2] =
			  rgb[3*((i-1)*width+(j-1))+2]*filter[0]
			+ rgb[3*((i-1)*width+j)+2]*filter[1]
			+ rgb[3*((i-1)*width+(j+1))+2]*filter[2]
			+ rgb[3*(i*width+(j-1))+2]*filter[3]
			+ rgb[3*(i*width+j)+2]*filter[4]
			+ rgb[3*(i*width+(j+1))+2]*filter[5]
			+ rgb[3*((i+1)*width+(j-1))+2]*filter[6]
			+ rgb[3*((i+1)*width+j)+2]*filter[7]
			+ rgb[3*((i+1)*width+(j+1))+2]*filter[8]+128;
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
	
	vector<unsigned char> vect(m_in.rows*m_in.cols*3);
	Mat m_out(m_in.rows,m_in.cols,CV_8UC3,vect.data());
	
	unsigned char* rgb = nullptr;
	cudaMallocHost(&rgb,vect.size());
	rgb = m_in.data;
	
	unsigned char* mA_dest=nullptr;
	unsigned char* mA_src=nullptr;

	int nb_stream = 8;
	
	cudaStream_t streams[nb_stream];
	for (int i=0;i<nb_stream;i++){
		cudaStreamCreate(&streams[i]);
	}
	
	cudaDeviceProp prop;
	cudaErrorIdentifier = cudaGetDeviceProperties(&prop, 0);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à la récupération des propriétés du device"<<endl;

	cudaErrorIdentifier = cudaMalloc(&mA_dest,vect.size()*sizeof(unsigned char));
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à l'allocation de mA_dest"<<endl;

	cudaErrorIdentifier = cudaMalloc(&mA_src,m_in.rows*m_in.cols*3*sizeof(unsigned char));
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à l'allocation de mA_src"<<endl;
	
	int section_size = vect.size()/nb_stream;
	for (int i=0;i<nb_stream;i++){
		cudaErrorIdentifier = cudaMemcpyAsync(mA_src+section_size*i,rgb+section_size*i,section_size*sizeof(unsigned char),cudaMemcpyHostToDevice,streams[i]);
		if (cudaErrorIdentifier != cudaSuccess)
			cout<<"Erreur à l'envoi des données à mA_src sur le stream "<<i<<endl;
	}
	dim3 block(sqrt(prop.maxThreadsPerBlock), sqrt(prop.maxThreadsPerBlock));
	dim3 grid((m_in.cols + block.x - 1) / block.x, (m_in.rows + block.y - 1) / block.y);
	
	for (int i=0;i<nb_stream;i++){
		embossing<<<grid,block,0,streams[i]>>>(mA_dest,mA_src,m_in.cols,m_in.rows);
		cudaErrorIdentifier = cudaGetLastError();
		if (cudaErrorIdentifier != cudaSuccess){
			cout<<"Erreur à l'exécution de la fonction embossing"<<endl;
			cout<<"détail : "<<cudaGetErrorString(cudaErrorIdentifier)<<endl;
		}
	}
	
	cudaErrorIdentifier = cudaGetLastError();
	if (cudaErrorIdentifier != cudaSuccess){
		cout<<"Erreur à l'exécution de la fonction embossing"<<endl;
		cout<<"détail : "<<cudaGetErrorString(cudaErrorIdentifier)<<endl;
	}
	
	for (int i=0;i<nb_stream;i++){
		cudaErrorIdentifier = cudaMemcpyAsync(vect.data()+i*section_size,mA_dest+i*section_size,section_size*sizeof(unsigned char),cudaMemcpyDeviceToHost,streams[i]);
		if (cudaErrorIdentifier != cudaSuccess)
			cout<<"Erreur à la réception des données de mA_dest sur le stream "<<i<<endl;
	}
	cudaDeviceSynchronize();
	
	std::string path(argv[1]);
	imwrite( "out_embossing_stream_cu_"+path, m_out );
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
	
	for (int i=0;i<nb_stream;i++){
		cudaStreamDestroy(streams[i]);
	}
	
	cudaFree( mA_dest );
	cudaFree( mA_src );
 
	cudaFreeHost( rgb );
	
	return 0;
}




















