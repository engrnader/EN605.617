CXX = g++
CXXFLAGS = -std=c++11 -I.
LDFLAGS = -lOpenCL

all: assignment

assignment: assignment.cpp info.hpp assignment.cl
	$(CXX) $(CXXFLAGS) assignment.cpp -o assignment $(LDFLAGS)

clean:
	rm -f assignment assignment.exe

.PHONY: all clean
