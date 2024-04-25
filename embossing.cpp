#include <opencv2/opencv.hpp>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>

using namespace std;
using namespace cv;
using namespace chrono;

int main(int argc, char *argv[]){
	auto start = high_resolution_clock::now();
	Mat m_in = imread(argv[1], IMREAD_UNCHANGED );
	unsigned char* rgb = m_in.data;
	
	vector< unsigned char >vect(m_in.rows * m_in.cols*3);
	Mat m_out( m_in.rows, m_in.cols, CV_8UC3, vect.data() );
	vector<int> filter{0,0,-1,0,0,0,1,0,0};
	
	for (int i=3;i<3*(m_in.rows-1);i+=3){
		for (int j=3;j<3*(m_in.cols-1);j+=3){
			vect[i*m_in.cols+j] =
				//rgb[i*m_in.cols+j];
				rgb[(i-3)*m_in.cols+(j-3)]*filter[0]+
				rgb[(i-3)*m_in.cols+j]*filter[1]+
				rgb[(i-3)*m_in.cols+(j+3)]*filter[2]+
				rgb[i*m_in.cols+(j-3)]*filter[3]+
				rgb[i*m_in.cols+j]*filter[4]+
				rgb[i*m_in.cols+(j+3)]*filter[5]+
				rgb[(i+3)*m_in.cols+(j-3)]*filter[6]+
				rgb[(i+3)*m_in.cols+j]*filter[7]+
				rgb[(i+3)*m_in.cols+(j+3)]*filter[8]+128;
			vect[i*m_in.cols+j+1] =
				//rgb[i*m_in.cols+j+1];
				rgb[(i-3)*m_in.cols+(j-3)+1]*filter[0]+
				rgb[(i-3)*m_in.cols+j+1]*filter[1]+
				rgb[(i-3)*m_in.cols+(j+3)+1]*filter[2]+
				rgb[i*m_in.cols+(j-3)+1]*filter[3]+
				rgb[i*m_in.cols+j+1]*filter[4]+
				rgb[i*m_in.cols+(j+3)+1]*filter[5]+
				rgb[(i+3)*m_in.cols+(j-3)+1]*filter[6]+
				rgb[(i+3)*m_in.cols+j+1]*filter[7]+
				rgb[(i+3)*m_in.cols+(j+3)+1]*filter[8]+128;
			vect[i*m_in.cols+j+2] =
				//rgb[i*m_in.cols+j+2];
				rgb[(i-3)*m_in.cols+(j-3)+2]*filter[0]+
				rgb[(i-3)*m_in.cols+j+2]*filter[1]+
				rgb[(i-3)*m_in.cols+(j+3)+2]*filter[2]+
				rgb[i*m_in.cols+(j-3)+2]*filter[3]+
				rgb[i*m_in.cols+j+2]*filter[4]+
				rgb[i*m_in.cols+(j+3)+2]*filter[5]+
				rgb[(i+3)*m_in.cols+(j-3)+2]*filter[6]+
				rgb[(i+3)*m_in.cols+j+2]*filter[7]+
				rgb[(i+3)*m_in.cols+(j+3)+2]*filter[8]+128;
		}
	}
	std::string path(argv[1]);
	imwrite( "out_embossing_"+path, m_out );
	auto end = high_resolution_clock::now();
	const duration<double> temps = end - start;
	cout<<temps.count() << " secondes" << endl;
	cout<<temps.count()*1000 << " millisecondes" << endl;
	return 0;
}























