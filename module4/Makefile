all: assignment.exe

assignment.exe: assignment.cu
	nvcc -lineinfo assignment.cu -o assignment.exe

run: assignment.exe
	assignment.exe 64
	assignment.exe 128
	assignment.exe 256
	assignment.exe 512

clean:
	del assignment.exe
