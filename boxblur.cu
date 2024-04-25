#include <opencv2/opencv.hpp>

#include <vector>
#include <chrono>

using namespace std;
using namespace cv;

__global__ void boxblur(size_t rows, size_t cols, unsigned char*rgb, unsigned char* g)
{
	auto i = blockIdx.y * blockDim.y + threadIdx.y;
	auto j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > 0 && i < rows && j % cols != 0 && j % cols != cols-1){
		g[i*cols*3+j*3] =
			( rgb[3*((i-1)*cols+(j-1))]
			+ rgb[3*((i-1)*cols+j)]
			+ rgb[3*((i-1)*cols+(j+1))]
			+ rgb[3*(i*cols+(j-1))]
			+ rgb[3*(i*cols+j)]
			+ rgb[3*(i*cols+(j+1))]
			+ rgb[3*((i+1)*cols+(j-1))]
			+ rgb[3*((i+1)*cols+j)]
			+ rgb[3*((i+1)*cols+(j+1))])/9;
		g[i*cols*3+j*3+1] =
			( rgb[3*((i-1)*cols+(j-1))+1]
			+ rgb[3*((i-1)*cols+j)+1]
			+ rgb[3*((i-1)*cols+(j+1))+1]
			+ rgb[3*(i*cols+(j-1))+1]
			+ rgb[3*(i*cols+j)+1]
			+ rgb[3*(i*cols+(j+1))+1]
			+ rgb[3*((i+1)*cols+(j-1))+1]
			+ rgb[3*((i+1)*cols+j)+1]
			+ rgb[3*((i+1)*cols+(j+1))+1])/9;
		g[i*cols*3+j*3+2] =
			( rgb[3*((i-1)*cols+(j-1))+2]
			+ rgb[3*((i-1)*cols+j)+2]
			+ rgb[3*((i-1)*cols+(j+1))+2]
			+ rgb[3*(i*cols+(j-1))+2]
			+ rgb[3*(i*cols+j)+2]
			+ rgb[3*(i*cols+(j+1))+2]
			+ rgb[3*((i+1)*cols+(j-1))+2]
			+ rgb[3*((i+1)*cols+j)+2]
			+ rgb[3*((i+1)*cols+(j+1))+2])/9;
	}
}

int main(int argc, char *argv[])
{
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
	unsigned char* rgb = m_in.data;
	unsigned char* rgb_c = nullptr;

	size_t const rows = m_in.rows;
	size_t const cols = m_in.cols;
	size_t const size = rows * cols;

	cudaDeviceProp prop;
	cudaErrorIdentifier = cudaGetDeviceProperties(&prop, 0);
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à la récupération des propriétés du device"<<endl;

	vector< unsigned char > g( size*3 );
	Mat m_out( rows, cols, CV_8UC3, g.data() );
	unsigned char* out_boxblur = nullptr;
	
	cudaErrorIdentifier = cudaMalloc( &out_boxblur, 3 * size * sizeof( unsigned char ) );
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à l'allocation de out_boxblur"<<endl;

	cudaErrorIdentifier = cudaMalloc( &rgb_c, 3 * size * sizeof( unsigned char ) );
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à l'allocation de rgb_c"<<endl;
	
	cudaErrorIdentifier = cudaMemcpy(rgb_c, rgb, 3 * size * sizeof( unsigned char ), cudaMemcpyHostToDevice );
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à l'envoi des données à rgb_c"<<endl;
			
	cudaErrorIdentifier = cudaMemcpy(out_boxblur, m_in.data, 3 * size * sizeof( unsigned char ), cudaMemcpyHostToDevice );
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à l'envoi des données à out_boxblur"<<endl;
		
	dim3 block(sqrt(prop.maxThreadsPerBlock), sqrt(prop.maxThreadsPerBlock));
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
	boxblur<<<grid,block>>>(rows,cols,rgb_c,out_boxblur);
	cudaErrorIdentifier = cudaGetLastError();
	if (cudaErrorIdentifier != cudaSuccess){
		cout<<"Erreur à l'exécution de la fonction boxblur"<<endl;
		cout<<"détail : "<<cudaGetErrorString(cudaErrorIdentifier)<<endl;
	}
	cudaErrorIdentifier = cudaMemcpy(g.data(), out_boxblur, g.size() * sizeof( unsigned char ), cudaMemcpyDeviceToHost );
	if (cudaErrorIdentifier != cudaSuccess)
		cout<<"Erreur à la réception des données de out_boxblur"<<endl;

	std::string path(argv[1]);
	imwrite( "out_boxblur_cu_"+path, m_out );

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
	
	return 0;
}
