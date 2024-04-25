#include <opencv2/opencv.hpp>

#include <vector>
#include <chrono>

using namespace std;
using namespace chrono;

int main(int argc, char *argv[])
{
  auto start = high_resolution_clock::now();
  cv::Mat m_in = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
  unsigned char *rgb = m_in.data;

  int cols = m_in.cols;
  int rows = m_in.rows;

  vector<unsigned char> g(rows * cols * 3);
  cv::Mat m_out(rows, cols, CV_8UC3, g.data());

  for (int j = 3; j < 3 *(rows - 1); j+=3)
  {
    for (int i = 3; i < 3 *(cols - 1); i+=3)
    {
        auto t =
            ( rgb[((j - 3) * m_in.cols + i - 3)] 
            + rgb[((j - 3) * m_in.cols + i)] 
            + rgb[((j - 3) * m_in.cols + i + 3)] 
            + rgb[(j * m_in.cols + i - 3)] 
            + rgb[(j * m_in.cols + i)] 
            + rgb[(j * m_in.cols + i + 3)] 
            + rgb[((j + 3) * m_in.cols + i - 3)] 
            + rgb[((j + 3) * m_in.cols + i)] 
            + rgb[((j + 3) * m_in.cols + i + 3)])/9;
        g[j * m_in.cols + i] = t;
        auto t1 =
            ( rgb[((j - 3) * m_in.cols + i - 3)+1] 
            + rgb[((j - 3) * m_in.cols + i)+1] 
            + rgb[((j - 3) * m_in.cols + i + 3)+1] 
            + rgb[(j * m_in.cols + i - 3)+1] 
            + rgb[(j * m_in.cols + i)+1] 
            + rgb[(j * m_in.cols + i + 3)+1] 
            + rgb[((j + 3) * m_in.cols + i - 3)+1] 
            + rgb[((j + 3) * m_in.cols + i)+1] 
            + rgb[((j + 3) * m_in.cols + i + 3)+1])/9;
        g[j * m_in.cols + i+1] = t1;
        auto t2 =
            ( rgb[((j - 3) * m_in.cols + i - 3)+2] 
            + rgb[((j - 3) * m_in.cols + i)+2] 
            + rgb[((j - 3) * m_in.cols + i + 3)+2] 
            + rgb[(j * m_in.cols + i - 3)+2] 
            + rgb[(j * m_in.cols + i)+2] 
            + rgb[(j * m_in.cols + i + 3)+2] 
            + rgb[((j + 3) * m_in.cols + i - 3)+2] 
            + rgb[((j + 3) * m_in.cols + i)+2] 
            + rgb[((j + 3) * m_in.cols + i + 3)+2])/9;
        g[j * m_in.cols + i+2] = t2;
    }
  }
  std::string path(argv[1]);
  cv::imwrite("out_boxblur_"+path, m_out);
  auto end = high_resolution_clock::now();
  const duration<double> temps = end - start;
  cout << temps.count() << " secondes" << endl;
  cout << temps.count()*1000 << " millisecondes" << endl;
  return 0;
}
