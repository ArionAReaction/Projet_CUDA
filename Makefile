CXX=g++
CXXFLAGS=-O3 -march=native -I/usr/include/opencv4/
LDLIBS1=`pkg-config --libs opencv4`
NVFLAGS=-O3 -I/usr/include/opencv4/ -ccbin g++-10

all: boxblur boxblur-cu embossing embossing-cu embossing-stream gaussian_blur gaussian_blur-cu sobel sobel-cu

boxblur: boxblur.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS1)

boxblur-cu: boxblur.cu
	nvcc $(NVFLAGS) -o $@ $<  $(LDLIBS1)

embossing: embossing.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS1)
	
embossing-cu: embossing.cu
	nvcc $(NVFLAGS) -o $@ $<  $(LDLIBS1)

embossing-stream: embossing_stream.cu
	nvcc $(NVFLAGS) -o $@ $<  $(LDLIBS1)

gaussian_blur: gaussian_blur.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS1)

gaussian_blur-cu: gaussian_blur.cu
	nvcc $(NVFLAGS) -o $@ $<  $(LDLIBS1)

sobel: sobel.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS1)

sobel-cu: sobel.cu
	nvcc $(NVFLAGS) -o $@ $<  $(LDLIBS1)

.PHONY: clean

clean:
	rm boxblur boxblur-cu embossing out* embossing-cu embossing-stream gaussian_blur gaussian_blur-cu sobel sobel-cu 
