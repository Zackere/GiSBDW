#include <iostream>
#include <cuda_runtime.h>

__global__ void asd() {
  printf("ASD\n");
  __syncthreads();
  return;
}

int main() {
  std::cout << "Hellow\n";
  asd<<<1, 2>>>();
  return 0;
}