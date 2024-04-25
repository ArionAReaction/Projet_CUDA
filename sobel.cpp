#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

using namespace std;
using namespace cv;
using namespace chrono;

int main(int argc, char *argv[]){
	auto start = high_resolution_clock::now();
	Mat m_in = imread(argv[1], IMREAD_UNCHANGED );
	auto rgb = m_in.data;
	
	vector<unsigned char> g( m_in.rows * m_in.cols);
	Mat m_out( m_in.rows, m_in.cols, CV_8UC1, g.data() );
	
	vector< unsigned char > inter( m_in.rows * m_in.cols);
	for (int i=0;i<m_in.rows;i++){
		for (int j = 0;j<m_in.cols;j++){
			inter[i*m_in.cols+j] = (307*rgb[3*(i*m_in.cols+j)]+604*rgb[3*(i*m_in.cols+j)+1]+113*rgb[3*(i*m_in.cols+j)+2])/1024;
		}
	}
	int h,v,res;

	
	for(size_t j=1;j<m_in.rows-1;++j){
		for(size_t i=1;i<m_in.cols-1;++i){
			h=inter[(j-1)*m_in.cols+i-1]-inter[(j-1)*m_in.cols+i+1]
			+2*inter[j*m_in.cols+i-1]-2*inter[j*m_in.cols+i+1]
			+inter[(j+1)*m_in.cols+i-1]-inter[(j+1)*m_in.cols+i+1];

			v=inter[(j-1)*m_in.cols+i-1]-inter[(j+1)*m_in.cols+i-1]
			+2*inter[(j-1)*m_in.cols+i]-2*inter[(j+1)*m_in.cols+i]
			+inter[(j-1)*m_in.cols+i+1]-inter[(j+1)*m_in.cols+i+1];

			res = h*h + v*v;
			res = res > 255*255 ? res = 255*255 : res;
			g[j*m_in.cols+i] = sqrt(res);
		}
	}
	
	std::string path(argv[1]);
	imwrite( "out_sobel_"+path, m_out );
	auto end = high_resolution_clock::now();
	const duration<double> temps = end - start;
	cout<<temps.count() << " secondes" << endl;
	cout<<temps.count()*1000 << " millisecondes" << endl;
	return 0;
}
