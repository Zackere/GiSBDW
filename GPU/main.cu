#include <cuda_runtime.h>

#include <iostream>

__global__ void asd() {
  printf("ASD\n");
  __syncthreads();
  return;
}

int main() {
  std::cout << "Hellow\n";
  asd<<<1, 2>>>();
  cudaStreamSynchronize(0);
  return 0;
}
