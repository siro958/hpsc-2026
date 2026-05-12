#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void bucket_count(int* key, int* bucket, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    atomicAdd(&bucket[key[idx]], 1);
  }
}

__global__ void calc_offset(int* bucket, int* offset, int range) {
  if(threadIdx.x == 0 && blockIdx.x == 0){
    offset[0] = 0;
    for(int i = 1; i < range; i++){
      offset[i] = offset[i-1] + bucket[i-1];
    }
  }
}

__global__ void set_keys(int* key, int* bucket, int* offset, int range) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < range) {
    for (int i=0; i<bucket[idx]; i++) {
      key[offset[idx] + i] = idx;
    }
  }
}

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int *d_key, *d_bucket, *d_offset;
  cudaMalloc(&d_key, n*sizeof(int));
  cudaMalloc(&d_bucket, range*sizeof(int));
  cudaMalloc(&d_offset, range*sizeof(int));
  cudaMemcpy(d_key, key.data(), n*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(d_bucket, 0, range*sizeof(int));

  bucket_count<<<(n+255)/256, 256>>>(d_key, d_bucket, n);
  calc_offset<<<1, 1>>>(d_bucket, d_offset, range);
  set_keys<<<(range+255)/256, 256>>>(d_key, d_bucket, d_offset, range);

  cudaMemcpy(key.data(), d_key, n*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(d_key);
  cudaFree(d_bucket);
  cudaFree(d_offset);
}
